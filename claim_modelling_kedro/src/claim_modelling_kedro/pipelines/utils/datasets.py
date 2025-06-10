import importlib
import logging
import os
import tempfile
import threading
import time
from typing import Dict, Any, Optional

import mlflow
import pandas as pd
from kedro.io import DataCatalog

from claim_modelling_kedro.pipelines.p01_init.config import get_subruns_dct, Config
from claim_modelling_kedro.pipelines.utils.utils import parse_catalog_as_dict, remove_file_or_directory

logger = logging.getLogger(__name__)


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


def get_partition(df_dct: Dict[str, Any], partition: str) -> Any:
    part_df = df_dct[partition]
    if callable(part_df):
        part_df = part_df()
    return part_df


def get_mlflow_run_id_for_partition(config: Config, partition: str, parent_mlflow_run_id: str = None) -> str:
    if parent_mlflow_run_id is not None:
        parent_run = mlflow.get_run(parent_mlflow_run_id)
    else:
        parent_run = mlflow.active_run()
    logger.debug(f"parent_run: {parent_run}, run_id: {parent_run.info.run_id}")
    subruns_dct = get_subruns_dct(parent_run) or (config.data.cv_mlflow_runs_ids if config is not None else None)
    logger.debug(f"subruns_dct: {subruns_dct}")
    if subruns_dct is not None:
        return subruns_dct[partition]
    else:
        return parent_run.info.run_id


def save_partitioned_dataset_in_mlflow(df: Dict[str, pd.DataFrame], artifact_path: str, keys: Dict[str, pd.Index] = None):
    for part in df.keys():
        df_part = get_partition(df, part)
        if keys is not None:
            keys_part = get_partition(keys, part)
            df_part = df_part.loc[keys_part, :]
        with tempfile.TemporaryDirectory() as temp_dir:
            pq_path = os.path.join(temp_dir, f"{part}.pq")
            df_part.to_parquet(pq_path, index=True)
            mlflow.log_artifact(pq_path, artifact_path=artifact_path)


def load_partitioned_dataset_from_mlflow(
    artifact_path: str,
    mlflow_run_id: Optional[str] = None,
    time_limit: Optional[float] = None,
    raise_on_failure: bool = True
) -> Optional[Dict[str, pd.DataFrame]]:

    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id

    try:
        artifacts = mlflow.artifacts.list_artifacts(run_id=mlflow_run_id, artifact_path=artifact_path)
        partitioned_dataset = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            for artifact in artifacts:
                artifact_uri = f"runs:/{mlflow_run_id}/{artifact.path}"
                result = {}
                exception = {}

                def download_partition():
                    try:
                        local_path = mlflow.tracking.artifact_utils._download_artifact_from_uri(artifact_uri, temp_dir)
                        df = pd.read_parquet(local_path)
                        result["df"] = df
                    except Exception as e:
                        exception["error"] = e

                thread = threading.Thread(target=download_partition)
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
                    msg = f"Error loading artifact {artifact.path}: {exception['error']}"
                    if raise_on_failure:
                        logger.error(msg, exc_info=True)
                        raise exception["error"]
                    else:
                        logger.error(msg)
                        return None

                partition_name = os.path.splitext(os.path.basename(artifact.path))[0]
                partitioned_dataset[partition_name] = result["df"]

        return partitioned_dataset

    except Exception as e:
        msg = f"Failed to load partitioned dataset from MLflow: {e}"
        if raise_on_failure:
            logger.error(msg, exc_info=True)
            raise
        else:
            logger.error(msg)
            return None

