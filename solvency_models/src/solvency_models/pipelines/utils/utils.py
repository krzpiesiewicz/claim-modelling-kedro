import importlib
import logging
import shutil
import tempfile
from typing import Sequence, Union, Dict, Any, Tuple

from kedro.io import DataCatalog
import mlflow
import numpy as np
import pandas as pd

from solvency_models.pipelines.p01_init.config import get_subruns_dct, Config

logger = logging.getLogger(__name__)

import yaml
import os


def parse_catalog_as_dict(catalog_path=None):
    """
    Function to parse the Kedro catalog.yml as a dictionary.

    Args:
        catalog_path (str): The path to the catalog.yml file.
                            If None, defaults to 'conf/base/catalog.yml'.

    Returns:
        dict: Parsed catalog.yml content as a dictionary.
    """
    if catalog_path is None:
        catalog_path = os.path.join("conf", "base", "catalog.yml")

    # Check if the catalog.yml file exists
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog file not found at {catalog_path}")

    # Load the catalog.yml file
    with open(catalog_path, 'r') as file:
        catalog_dict = yaml.safe_load(file)

    return catalog_dict


def remove_file_or_directory(file_path):
    """
    Check if the file or directory exists, and remove it.

    Args:
        file_path (str): The path to the file or directory to be removed.
    """
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            # If it's a directory, remove the directory and all its contents
            shutil.rmtree(file_path)
            logger.info(f"Directory '{file_path}' removed.")
        else:
            # If it's a file, remove the file
            os.remove(file_path)
            logger.info(f"File '{file_path}' removed.")
    else:
        logger.info(f"File or directory '{file_path}' does not exist.")


def remove_dataset(catalog: DataCatalog, dataset_name: str):
    # Check if the dataset exists in the catalog
    if dataset_name not in catalog.list():
        logger.warning(f"Dataset '{dataset_name}' not found in the catalog.")
        return

    logger.info(f"Removing dataset '{dataset_name}'...")

    # Get the dataset object
    catalog_dict = parse_catalog_as_dict()
    dataset_dict = catalog_dict[dataset_name]

    # Check if the dataset has a filepath (for file-based datasets)
    if "path" in dataset_dict:
        file_path = dataset_dict["path"]
        remove_file_or_directory(file_path)
    else:
        logger.info(f"Dataset '{dataset_name}' is not file-based or does not have a file path.")

    dataset = catalog._get_dataset(dataset_name)
    dataset.release()
    logger.info(f"Dataset '{dataset_name}' released.")


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


def get_mlflow_run_id_for_partition(config: Config, partition: str, parent_mflow_run_id: str = None) -> str:
    if parent_mflow_run_id is not None:
        parent_run = mlflow.get_run(parent_mflow_run_id)
    else:
        parent_run = mlflow.active_run()
    logger.debug(f"parent_run: {parent_run}, run_id: {parent_run.info.run_id}")
    subruns_dct = get_subruns_dct(parent_run) or (config.data.cv_mlflow_runs_ids if config is not None else None)
    logger.debug(f"subruns_dct: {subruns_dct}")
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
            logger.info(f"Loading artifact {artifact.path} from MLFlow...")
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


def load_predictions_and_target_from_mlflow(dataset: str,
                                            mlflow_run_id: str = None) -> Tuple[Dict[str, pd.DataFrame]]:
    logger.info(f"Loading the predictions and the target from {dataset} dataset from MLFlow...")
    predictions_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/prediction",
                                                          mlflow_run_id=mlflow_run_id)
    target_df = load_partitioned_dataset_from_mlflow(artifact_path=f"predictions/{dataset}/target",
                                                     mlflow_run_id=mlflow_run_id)
    logger.info(f"Loaded the predictions and the target from {dataset} dataset from MLFlow.")
    return predictions_df, target_df


def save_pd_dataframe_as_csv_in_mlflow(df: pd.DataFrame, artifact_path: str, csv_filename: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, csv_filename)
        df = df.reset_index()
        df.columns.name = None
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        return csv_path


def load_pd_dataframe_csv_from_mlflow(artifact_path: str, mlflow_run_id: str = None, index_col=None,
                                      columns_index_name=None) -> pd.DataFrame:
    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id
    with tempfile.TemporaryDirectory() as temp_dir:
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
    _, artifact_path = get_metrics_table_filname_and_artifact_path(dataset, prefix)
    metrics_df = load_pd_dataframe_csv_from_mlflow(artifact_path, mlflow_run_id, index_col="metric",
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
    _, artifact_path = get_metrics_cv_stats_filname_and_artifact_path(dataset, prefix)
    metrics_cv_stats_df = load_pd_dataframe_csv_from_mlflow(artifact_path, mlflow_run_id, index_col="metric")
    return metrics_cv_stats_df


def convert_np_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(i) for i in obj]
    else:
        return obj
