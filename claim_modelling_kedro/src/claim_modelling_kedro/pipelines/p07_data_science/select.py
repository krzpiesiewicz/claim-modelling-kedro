import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import mlflow
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from claim_modelling_kedro.pipelines.utils.datasets import get_class_from_path, get_partition, get_mlflow_run_id_for_partition

logger = logging.getLogger(__name__)

# Define the artifact path for selector
_selector_artifact_path = "model_ds/selector"
_select_artifact_path = "model_ds"
_selected_features_filename = "selected_features.txt"
_features_importances_filename = "features_importances.csv"


class SelectorModel(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.max_n_features = config.ds.fs_max_n_features
        self.min_importance = config.ds.fs_min_importance
        self.max_iter = config.ds.fs_max_iter
        self._features_importances = None
        self._selected_features = None

    @abstractmethod
    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return features_df.loc[:, self._selected_features]

    def get_selected_features(self) -> pd.Index:
        return self._selected_features.copy()

    def get_features_importances(self) -> pd.Series:
        if self._features_importances is not None:
            return self._features_importances.copy()
        return None

    def _set_selected_features_from_importances(self):
        if self.min_importance is not None:
            importances = self._features_importances[self._features_importances > self.min_importance]
        else:
            importances = self._features_importances.index
        if self.max_n_features is not None:
            importances = importances.sort_values(ascending=False).head(self.max_n_features)
        self._set_selected_features(importances.index)

    def _set_selected_features(self, selected_features: Sequence[str]):
        self._selected_features = pd.Index(selected_features, name="feature")

    def _set_features_importances(self, features_importances: pd.Series):
        features_importances.index.name = "feature"
        features_importances.name = "importance"
        self._features_importances = features_importances.sort_values(ascending=False)


def process_select_features_partition(config: Config, part: str, transformed_features_df: Dict[str, pd.DataFrame],
                                      mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    logger.info(f"Selecting features for partition '{part}' by the MLFlow model...")
    features_part_df = get_partition(transformed_features_df, part)
    selector = MLFlowModelLoader("features selector model").load_model(path=_selector_artifact_path, run_id=mlflow_run_id)
    selected_features_part_df = selector.transform(features_part_df)
    logger.info(f"Selected features for partition '{part}'.")
    return part, selected_features_part_df

def select_features_by_mlflow_model(config: Config, transformed_features_df: Dict[str, pd.DataFrame],
                                    mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    logger.info("Selecting features by the MLFlow model...")
    selected_features_df = {}

    parts_cnt = len(transformed_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_select_features_partition, config, part, transformed_features_df, mlflow_run_id): part
            for part in transformed_features_df.keys()
        }
        for future in as_completed(futures):
            part, selected_features_part_df = future.result()
            selected_features_df[part] = selected_features_part_df

    logger.info("Selected features for all partitions.")
    return selected_features_df


def fit_transform_features_selector_part(config: Config, transformed_sample_features_df: pd.DataFrame,
                                         sample_target_df: pd.DataFrame, sample_train_keys: pd.Index,
                                         sample_val_keys: pd.Index) -> pd.DataFrame:
    selector = get_class_from_path(config.ds.fs_model_class)(config=config)
    logger.info("Fitting the features selector...")
    selector.fit(transformed_sample_features_df.loc[sample_train_keys,:], sample_target_df.loc[sample_train_keys,:])
    logger.info("Fitted the features selector.")
    # Save the selector
    MLFlowModelLogger(selector, f"features selector model").log_model(_selector_artifact_path)
    selected_features = selector.get_selected_features()
    features_importances = selector.get_features_importances()
    # Log info about the features importances and selected features
    logger.info(f"Features importances:\n{features_importances.reset_index()}")
    logger.info(f"Selected {len(selected_features)} / {len(features_importances)} features:\n" + \
                ",\n".join(f"    - {ftr}" for ftr in selected_features))
    sel_ftrs_imp = features_importances[selected_features]
    logger.info(f"""Importances of the selected features:
    - min: {sel_ftrs_imp.min()}
    - mean: {sel_ftrs_imp.mean()}
    - max: {sel_ftrs_imp.max()}""")
    # Save the selected features
    logger.info("Saving the selected features...")
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, _selected_features_filename)
        with open(file_path, "w") as f:
            content = "\n".join(selected_features.tolist())
            f.write(content)
        mlflow.log_artifact(file_path, _select_artifact_path)
    logger.info(
        f"Successfully saved the selected features to MLFlow path {os.path.join(_select_artifact_path, _selected_features_filename)}")
    # Save the features importances
    logger.info("Saving the features importances...")
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, _features_importances_filename)
        features_importances.to_csv(file_path)
        mlflow.log_artifact(file_path, _select_artifact_path)
    logger.info(
        f"Successfully saved the features importances to MLFlow path {os.path.join(_select_artifact_path, _features_importances_filename)}")
    selected_features_df = selector.transform(transformed_sample_features_df)
    logger.info("Selected the features.")
    return selected_features_df


def process_fit_transform_partition(config: Config, part: str, sample_features_df: Dict[str, pd.DataFrame],
                                    sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                                    sample_val_keys: Dict[str, pd.Index]) -> Tuple[str, pd.DataFrame]:
    features_part_df = get_partition(sample_features_df, part)
    target_part_df = get_partition(sample_target_df, part)
    sample_train_keys_part = get_partition(sample_train_keys, part)
    sample_val_keys_part = get_partition(sample_val_keys, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
    logger.info(f"Fitting selector on partition '{part}' of the sample dataset...")
    with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
        selected_part_df = fit_transform_features_selector_part(config, features_part_df, target_part_df,
                                                                sample_train_keys_part, sample_val_keys_part)
    return part, selected_part_df


def fit_transform_features_selector(config: Config, sample_features_df: Dict[str, pd.DataFrame],
                                    sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                                    sample_val_keys: Dict[str, pd.Index]) -> Dict[str, pd.DataFrame]:
    logger.info(f"Fitting features selector on the sample dataset...")
    selected_features_df = {}

    parts_cnt = len(sample_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_fit_transform_partition, config, part, sample_features_df, sample_target_df,
                            sample_train_keys, sample_val_keys): part
            for part in sample_features_df.keys()
        }
        for future in as_completed(futures):
            part, selected_part_df = future.result()
            selected_features_df[part] = selected_part_df

    return selected_features_df


def process_select_partition(config: Config, part: str, features_df: Dict[str, pd.DataFrame],
                             mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    features_part_df = get_partition(features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
    logger.info(f"Selecting partition '{part}' of the sample dataset...")
    selected_part_df = select_features_by_mlflow_model_part(config, features_part_df, mlflow_subrun_id)
    return part, selected_part_df


def select_features_by_mlflow_model(config: Config, features_df: Dict[str, pd.DataFrame],
                                    mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    if not config.ds.fs_enabled:
        logger.info("Feature selection is disabled. Hence, returning the input features.")
        return features_df
    selected_features_df = {}
    logger.info(f"Selecting features of the sample dataset...")

    parts_cnt = len(features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_select_partition, config, part, features_df, mlflow_run_id): part
            for part in features_df.keys()
        }
        for future in as_completed(futures):
            part, selected_part_df = future.result()
            selected_features_df[part] = selected_part_df

    return selected_features_df
