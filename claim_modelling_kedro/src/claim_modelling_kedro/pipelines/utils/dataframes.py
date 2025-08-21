import logging
import os
import tempfile
import threading
from typing import Union, Dict, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
import hashlib

from claim_modelling_kedro.pipelines.utils.datasets import save_partitioned_dataset_in_mlflow, \
    load_partitioned_dataset_from_mlflow

logger = logging.getLogger(__name__)


def stable_str_hash(x: int, seed: str = "") -> str:
    """
    Deterministic and unique string hash for an integer input.

    Args:
        x (int): The integer to hash.
        seed (str): Optional string to introduce variability across runs.

    Returns:
        str: Hexadecimal SHA256 hash of the input.
    """
    s = f"{seed}-{x}"
    return hashlib.sha256(s.encode()).hexdigest()


def indices_ordered_values_and_hashed_index(
    s: Union[pd.Series, np.ndarray],
) -> np.ndarray:
    # Sort inputs stably by y_pred, then by hashed index
    s = pd.Series(s)
    secondary_key = s.index.to_series().apply(lambda x: stable_str_hash(x)).values
    primary_key = s.values
    sorted_idx = np.lexsort((secondary_key, primary_key))
    return sorted_idx


def ordered_by_pred_and_hashed_index(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        sample_weight: Union[pd.Series, np.ndarray] = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    if sample_weight is None:
        sample_weight = pd.Series(np.ones_like(y_true), index=y_true.index)
    else:
        sample_weight = pd.Series(sample_weight)

    sorted_idx = indices_ordered_values_and_hashed_index(y_pred)

    y_true = y_true.iloc[sorted_idx].reset_index(drop=True)
    y_pred = y_pred.iloc[sorted_idx].reset_index(drop=True)
    sample_weight = sample_weight.iloc[sorted_idx].reset_index(drop=True)

    return y_true, y_pred, sample_weight


def index_ordered_by_col(
    df: pd.DataFrame,
    order_col: str,
) -> pd.Index:
    sorted_idx = indices_ordered_values_and_hashed_index(df[order_col])
    return df.index[sorted_idx]


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


def load_predictions_and_target_from_mlflow(
        dataset: str,
        mlflow_run_id: str = None,
        time_limit: float = None,
        raise_on_failure: bool = True
) -> Tuple[Dict[str, pd.DataFrame]]:
    logger.info(f"Loading the predictions and the target from {dataset} dataset from MLFlow...")
    predictions_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/prediction",
                                                          mlflow_run_id=mlflow_run_id, time_limit=time_limit,
                                                          raise_on_failure=raise_on_failure)
    target_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/target",
                                                     mlflow_run_id=mlflow_run_id, time_limit=time_limit,
                                                     raise_on_failure=raise_on_failure)
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


def load_pd_dataframe_csv_from_mlflow(
    artifact_path: str,
    filename: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    index_col=None,
    columns_index_name=None,
    time_limit: float = None,
    raise_on_failure: bool = True
) -> Optional[pd.DataFrame]:

    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id

    artifact_path = f"{artifact_path}/{filename}" if filename is not None else artifact_path
    artifact_uri = f"runs:/{mlflow_run_id}/{artifact_path}"

    result = {}
    exception = {}

    def download():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = mlflow.tracking.artifact_utils._download_artifact_from_uri(artifact_uri, temp_dir)
                df = pd.read_csv(local_path, index_col=index_col)
                df.columns.name = columns_index_name
                result["df"] = df
        except Exception as e:
            exception["error"] = e

    thread = threading.Thread(target=download)
    thread.start()
    thread.join(timeout=time_limit)

    if thread.is_alive():
        msg = f"Timeout: downloading artifact exceeded {time_limit} seconds: {artifact_uri}"
        if raise_on_failure:
            logger.error(msg)
            raise TimeoutError(msg)
        else:
            logger.error(msg)
            return None

    if "error" in exception:
        if raise_on_failure:
            logger.error("Failed to load DataFrame from MLflow", exc_info=True)
            raise exception["error"]
        else:
            logger.error(f"Failed to load artifact {artifact_uri}: {exception['error']}")
            return None

    return result.get("df")


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


def load_metrics_table_from_mlflow(
    dataset: str,
    prefix: str = None,
    mlflow_run_id: str = None,
    time_limit: float = None,
    raise_on_failure: bool = True
) -> pd.DataFrame:
    filename, artifact_path = get_metrics_table_filname_and_artifact_path(dataset, prefix)
    metrics_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id, index_col="metric",
                                                   columns_index_name="part", time_limit=time_limit,
                                                   raise_on_failure=raise_on_failure)
    return metrics_df


def get_metrics_cv_stats_filname_and_artifact_path(dataset: str, prefix: str = None) -> Tuple[str, str]:
    filename = _file_name(dataset, prefix, "metrics_cv_stats", "csv")
    artifact_path = f"results/{dataset}"
    return filename, artifact_path


def save_metrics_cv_stats_in_mlflow(metrics_cv_stats_df: pd.DataFrame, dataset: str, prefix: str = None):
    csv_filename, artifact_path = get_metrics_cv_stats_filname_and_artifact_path(dataset, prefix)
    save_pd_dataframe_as_csv_in_mlflow(metrics_cv_stats_df, artifact_path=artifact_path, csv_filename=csv_filename)


def load_metrics_cv_stats_from_mlflow(
    dataset: str,
    prefix: str = None,
    mlflow_run_id: str = None,
    time_limit: float = None,
    raise_on_failure: bool = True
) -> pd.DataFrame:
    filename, artifact_path = get_metrics_cv_stats_filname_and_artifact_path(dataset, prefix)
    metrics_cv_stats_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id, index_col="metric",
                                                            time_limit=time_limit, raise_on_failure=raise_on_failure)
    return metrics_cv_stats_df
