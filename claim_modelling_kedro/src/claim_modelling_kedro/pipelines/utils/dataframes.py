import logging
import os
import tempfile
from typing import Union, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.utils.datasets import save_partitioned_dataset_in_mlflow, \
    load_partitioned_dataset_from_mlflow

logger = logging.getLogger(__name__)


def assert_pandas_no_lacking_indexes(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     target_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     source_df_name: str,
                                     target_df_name: str):
    if type(source_df) is pd.DataFrame or type(source_df) is pd.Series:
        if not source_df.index.intersection(target_df.index).equals(source_df.index):
            raise ValueError(f"""{target_df_name} does not have some indexes from {source_df_name}:
    - {source_df_name}.index: {source_df.index}
    - {target_df_name}.index: {target_df.index}
    - lacking indexes: {source_df.index.difference(target_df.index)}""")


def trunc_target_index(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                       target_df: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    if type(source_df) is pd.DataFrame:
        target_df = target_df.loc[source_df.index, :]
    if type(source_df) is pd.Series:
        target_df = target_df.loc[source_df.index]
    return target_df


def preds_as_dataframe_with_col_name(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     preds: Union[pd.DataFrame, pd.Series, np.ndarray], pred_col: str) -> pd.DataFrame:
    if type(preds) is np.ndarray:
        if len(preds.shape) == 1:
            preds = pd.Series(preds)
        else:
            if preds.shape[0] != source_df.shape[0]:
                preds = preds.transpose()
            preds = pd.DataFrame(preds)
    if type(preds) is pd.Series:
        preds = pd.DataFrame(preds)
    if len(preds.columns) == 1:
        preds.columns = [pred_col]
    if not preds.index.equals(source_df.index):
        preds.set_index(source_df.index, inplace=True)
    return preds


def save_predictions_and_target_in_mlflow(predictions_df: Dict[str, pd.DataFrame], target_df: Dict[str, pd.DataFrame],
                                          dataset: str, keys: Dict[str, pd.Index] = None):
    logger.info(f"Saving the predictions and the target from {dataset} dataset to MLFlow...")
    save_partitioned_dataset_in_mlflow(predictions_df, artifact_path=f"predictions/{dataset}/prediction", keys=keys)
    save_partitioned_dataset_in_mlflow(target_df, artifact_path=f"predictions/{dataset}/target", keys=keys)
    logger.info(f"Saved the predictions and the target from {dataset} dataset to MLFlow.")


def load_predictions_and_target_from_mlflow(dataset: str,
                                            mlflow_run_id: str = None) -> Tuple[Dict[str, pd.DataFrame]]:
    logger.info(f"Loading the predictions and the target from {dataset} dataset from MLFlow...")
    predictions_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/prediction",
                                                          mlflow_run_id=mlflow_run_id)
    target_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/target",
                                                     mlflow_run_id=mlflow_run_id)
    logger.info(f"Loaded the predictions and the target from {dataset} dataset from MLFlow.")
    return predictions_df, target_df


def save_pd_dataframe_as_csv_in_mlflow(df: pd.DataFrame, artifact_path: str, csv_filename: str, index: bool = True):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, csv_filename)
        if index:
            df = df.reset_index()
        df.columns.name = None
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        return csv_path


def load_pd_dataframe_csv_from_mlflow(artifact_path: str, filename: str = None, mlflow_run_id: str = None,
                                      index_col=None,
                                      columns_index_name=None) -> pd.DataFrame:
    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = f"{artifact_path}/{filename}" if filename is not None else artifact_path
        artifact_uri = f"runs:/{mlflow_run_id}/{artifact_path}"
        local_path = mlflow.tracking.artifact_utils._download_artifact_from_uri(artifact_uri, temp_dir)
        df = pd.read_csv(local_path, index_col=index_col)
        df.columns.name = columns_index_name
        return df


def _file_name(dataset: str, prefix: str = None, suffix: str = None, ext: str = None) -> str:
    filename = f"{prefix}_{dataset}" if prefix is not None else dataset
    filename = f"{filename}_{suffix}" if suffix is not None else filename
    return f"{filename}.{ext}" if ext is not None else filename


def get_metrics_table_filname_and_artifact_path(dataset: str, prefix: str = None) -> Tuple[str, str]:
    filename = _file_name(dataset, prefix, "metrics_by_part", "csv")
    artifact_path = f"results/{dataset}"
    return filename, artifact_path


def save_metrics_table_in_mlflow(metrics_df: pd.DataFrame, dataset: str, prefix: str = None):
    csv_filename, artifact_path = get_metrics_table_filname_and_artifact_path(dataset, prefix)
    save_pd_dataframe_as_csv_in_mlflow(metrics_df, artifact_path=artifact_path, csv_filename=csv_filename)


def load_metrics_table_from_mlflow(dataset: str, prefix: str = None, mlflow_run_id: str = None) -> pd.DataFrame:
    filename, artifact_path = get_metrics_table_filname_and_artifact_path(dataset, prefix)
    metrics_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id, index_col="metric",
                                                   columns_index_name="part")
    return metrics_df


def get_metrics_cv_stats_filname_and_artifact_path(dataset: str, prefix: str = None) -> Tuple[str, str]:
    filename = _file_name(dataset, prefix, "metrics_cv_stats", "csv")
    artifact_path = f"results/{dataset}"
    return filename, artifact_path


def save_metrics_cv_stats_in_mlflow(metrics_cv_stats_df: pd.DataFrame, dataset: str, prefix: str = None):
    csv_filename, artifact_path = get_metrics_cv_stats_filname_and_artifact_path(dataset, prefix)
    save_pd_dataframe_as_csv_in_mlflow(metrics_cv_stats_df, artifact_path=artifact_path, csv_filename=csv_filename)


def load_metrics_cv_stats_from_mlflow(dataset: str, prefix: str = None, mlflow_run_id: str = None) -> pd.DataFrame:
    filename, artifact_path = get_metrics_cv_stats_filname_and_artifact_path(dataset, prefix)
    metrics_cv_stats_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id, index_col="metric")
    return metrics_cv_stats_df
