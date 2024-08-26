import importlib
import logging
import os
import tempfile
from typing import Sequence, Union, Dict, Any

import mlflow
import numpy as np
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config, get_subruns_dct

logger = logging.getLogger(__name__)


def get_class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_constructor = getattr(module, class_name)
    return class_constructor


def shortened_seq_to_string(seq: Sequence) -> str:
    def val_to_str(val):
        return f"{val:.3f}"

    lst = [val_to_str(elem) for elem in seq]
    if len(lst) <= 10:
        return ", ".join(lst)
    else:
        return ", ".join(lst[:5]) + " ... " + ", ".join(lst[-5:])


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


def get_partition(df_dct: Dict[str, Any], partition: str) -> Any:
    part_df = df_dct[partition]
    if callable(part_df):
        part_df = part_df()
    return part_df


def get_mlflow_run_id_for_partition(partition: str, parent_mflow_run_id: str = None) -> str:
    if parent_mflow_run_id is not None:
        parent_run = mlflow.get_run(parent_mflow_run_id)
    else:
        parent_run = mlflow.active_run()
    subruns_dct = get_subruns_dct(parent_run)
    if subruns_dct is not None:
        return subruns_dct[partition]
    else:
        return parent_run.info.run_id


def save_partitioned_dataset_in_mlflow(df: Dict[str, pd.DataFrame], artifact_path: str):
    for part in df.keys():
        df_part = get_partition(df, part)
        with tempfile.TemporaryDirectory() as temp_dir:
            pq_path = os.path.join(temp_dir, f"{part}.pq")
            df_part.to_parquet(pq_path, index=True)
            mlflow.log_artifact(pq_path, artifact_path=artifact_path)


def load_partitioned_dataset_from_mlflow(artifact_path: str, mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id

    # Get the list of artifact URIs in the artifact path
    artifacts = mlflow.artifacts.list_artifacts(run_id=mlflow_run_id, artifact_path=artifact_path)

    # Initialize an empty dictionary to hold the loaded partitions
    partitioned_dataset = {}

    # Create a temporary directory to download the artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download each artifact and load it as a DataFrame
        for artifact in artifacts:
            # Construct the artifact URI
            artifact_uri = f"runs:/{mlflow_run_id}/{artifact.path}"

            # Download the artifact to the temporary directory
            local_path = mlflow.tracking.artifact_utils._download_artifact_from_uri(artifact_uri, temp_dir)

            # Load the artifact as a DataFrame
            df = pd.read_parquet(local_path)

            # Get the partition name from the artifact path
            partition_name = os.path.splitext(os.path.basename(artifact.path))[0]

            # Add the DataFrame to the dictionary
            partitioned_dataset[partition_name] = df

    return partitioned_dataset


def save_predictions_and_target_in_mlflow(predictions_df: Dict[str, pd.DataFrame], target_df: Dict[str, pd.DataFrame],
                                          dataset: str):
    logger.info(f"Saving the predictions and the target from {dataset} dataset to MLFlow...")
    save_partitioned_dataset_in_mlflow(predictions_df, artifact_path=f"predictions/{dataset}/prediction")
    save_partitioned_dataset_in_mlflow(target_df, artifact_path=f"predictions/{dataset}/target")
    logger.info(f"Saved the predictions and the target from {dataset} dataset to MLFlow.")
