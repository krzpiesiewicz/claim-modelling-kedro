import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Sequence, Dict

import mlflow
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from solvency_models.pipelines.utils.utils import get_class_from_path, get_partition, get_mlflow_run_id_for_partition

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


def select_features_by_mlflow_model_part(config: Config, transformed_features_df: pd.DataFrame,
                                         mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info("Selecting features by the MLFlow model...")
    # Load the selector model from the MLflow model registry
    selector = MLFlowModelLoader("features selector model").load_model(path=_selector_artifact_path,
                                                                       run_id=mlflow_run_id)
    # Select the features
    selected_features_df = selector.transform(transformed_features_df)
    logger.info("Selected the features.")
    return selected_features_df


def fit_transform_features_selector_part(config: Config, transformed_sample_features_df: pd.DataFrame,
                                         sample_target_df: pd.DataFrame) -> pd.DataFrame:
    selector = get_class_from_path(config.ds.fs_model_class)(config=config)
    logger.info("Fitting the features selector...")
    selector.fit(transformed_sample_features_df, sample_target_df)
    logger.info("Fitted the features selector.")
    # Save the selector
    MLFlowModelLogger(selector, f"features selector model").log_model(_selector_artifact_path)
    # Save the selected features
    logger.info("Saving the selected features...")
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, _selected_features_filename)
        with open(file_path, "w") as f:
            content = "\n".join(selector.get_selected_features().tolist())
            f.write(content)
        mlflow.log_artifact(file_path, _select_artifact_path)
    logger.info(
        f"Successfully saved the selected features to MLFlow path {os.path.join(_select_artifact_path, _selected_features_filename)}")
    # Save the features importances
    logger.info("Saving the features importances...")
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, _features_importances_filename)
        selector.get_features_importances().to_csv(file_path)
        mlflow.log_artifact(file_path, _select_artifact_path)
    logger.info(
        f"Successfully saved the features importances to MLFlow path {os.path.join(_select_artifact_path, _features_importances_filename)}")
    selected_features_df = selector.transform(transformed_sample_features_df)
    logger.info("Selected the features.")
    return selected_features_df


def fit_transform_features_selector(config: Config, sample_features_df: Dict[str, pd.DataFrame],
                                    sample_target_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    logger.info(f"Fitting features selector on the sample dataset...")
    selected_features_df = {}
    for part in sample_features_df.keys():
        features_part_df = get_partition(sample_features_df, part)
        target_part_df = get_partition(sample_target_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(f"Fitting selector on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            selected_features_df[part] = fit_transform_features_selector_part(config, features_part_df, target_part_df)
    return selected_features_df


def select_features_by_mlflow_model(config: Config, features_df: Dict[str, pd.DataFrame],
                                    mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    selected_features_df = {}
    logger.info(f"Selecting features of the sample dataset...")
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
        logger.info(f"Selecting partition '{part}' of the sample dataset...")
        selected_features_df[part] = select_features_by_mlflow_model_part(config, features_part_df, mlflow_subrun_id)
    return selected_features_df
