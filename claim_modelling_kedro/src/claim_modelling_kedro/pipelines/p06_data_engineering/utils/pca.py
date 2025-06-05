import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import mlflow
import pandas as pd
from sklearn.decomposition import PCA

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader

logger = logging.getLogger(__name__)

_pca_artifact_path = "model_ds/pca"
_pca_model_name = "features pca model"


def get_pca_column_name(i: int, n: int) -> str:
    """
    Returns the name of a PCA column in the format "PC{i}", where i is zero-padded to match the length of the decimal representation of n.
    Args:
        i (int): Column number (1-indexed).
        n (int): Number determining the length of the decimal representation.

    Returns:
        str: Column name in the format "PC{i}".
    """
    width = len(str(n))
    return f"PC{i:0{width}d}"


def add_index_and_pc_columns(
        transformed_fearues: pd.DataFrame, index: pd.Index
) -> pd.DataFrame:
    """
    Adds index and PCA column names to the transformed features DataFrame.
    """
    columns = [get_pca_column_name(i, transformed_fearues.shape[1]) for i in range(transformed_fearues.shape[1])]
    return pd.DataFrame(transformed_fearues, index=index, columns=columns)


def process_features_pca_partition(config: Config, part: str, transformed_features_df: Dict[str, pd.DataFrame],
                                   mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    logger.info(f"Transforming features for partition '{part}' using PCA from MLFlow model...")
    features_part_df = get_partition(transformed_features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
    pca_model = MLFlowModelLoader(_pca_model_name).load_model(path=_pca_artifact_path, run_id=mlflow_subrun_id)
    transformed_df = pca_model.transform(features_part_df)
    transformed_df = add_index_and_pc_columns(transformed_df, index=features_part_df.index)
    logger.info(f"Transformed features for partition '{part}' with PCA.")
    return part, transformed_df


def transform_features_pca_by_mlflow_model(config: Config, transformed_features_df: Dict[str, pd.DataFrame],
                                           mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    logger.info("Transforming all partitions using PCA from MLFlow model...")
    transformed_data = {}
    parts_cnt = len(transformed_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_features_pca_partition, config, part, transformed_features_df, mlflow_run_id): part
            for part in transformed_features_df.keys()
        }
        for future in as_completed(futures):
            part, transformed_part = future.result()
            transformed_data[part] = transformed_part
    logger.info("Finished PCA transformation for all partitions.")
    return transformed_data


def fit_transform_features_pca_part(config: Config, transformed_sample_features_df: pd.DataFrame) -> pd.DataFrame:
    pca_model = PCA(n_components=config.de.pca_n_components)
    logger.info("Fitting PCA model...")
    pca_transformed_df = pca_model.fit_transform(transformed_sample_features_df, random_state=config.de.pca_random_state)
    pca_transformed_df = add_index_and_pc_columns(pca_transformed_df, index=transformed_sample_features_df.index)
    logger.info("Fitted PCA model.")
    # Log the PCA model to MLFlow
    MLFlowModelLogger(pca_model, _pca_model_name).log_model(_pca_artifact_path)
    logger.info("Logged PCA model to MLFlow.")
    return pca_transformed_df


def process_fit_transform_pca_partition(config: Config, part: str,
                                        sample_features_df: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    features_part_df = get_partition(sample_features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)

    logger.info(f"Fitting PCA model on partition '{part}'...")
    with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
        transformed_df = fit_transform_features_pca_part(config, features_part_df)
    return part, transformed_df


def fit_transform_features_pca(config: Config, sample_features_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    logger.info("Fitting PCA model on all sample partitions...")
    transformed_data = {}
    parts_cnt = len(sample_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_fit_transform_pca_partition, config, part, sample_features_df): part
            for part in sample_features_df.keys()
        }
        for future in as_completed(futures):
            part, transformed_part = future.result()
            transformed_data[part] = transformed_part
    logger.info("Finished fitting PCA models on all partitions.")
    return transformed_data
