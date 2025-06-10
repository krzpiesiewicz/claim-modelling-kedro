from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import mlflow
from typing import Dict, Tuple
import pandas as pd

from claim_modelling_kedro.pipelines.utils.datasets import (
    get_partition,
    get_mlflow_run_id_for_partition,
    get_class_from_path,
)
from claim_modelling_kedro.pipelines.utils.mlflow_model import (
    MLFlowModelLogger,
    MLFlowModelLoader,
)
from claim_modelling_kedro.pipelines.p01_init.config import Config


logger = logging.getLogger(__name__)

_target_transformer_artifact_path = "model_ds/target_transformer"


class TargetTransformer(ABC):
    """
    A class to transform target variables for machine learning models.
    """

    def __init__(self, target_col: str, pred_col: str = None):
        """
        Initializes the TargetTransformer with the specified target column and prediction column.

        Args:
            target_col (str): The name of the target column to transform.
            pred_col (str, optional): The name of the prediction column. Defaults to None.
        """
        self.target_col = target_col
        self.pred_col = pred_col

    @abstractmethod
    def _fit(self, y_true: pd.Series):
        ...

    @abstractmethod
    def _transform(self, y_true: pd.Series) -> pd.DataFrame:
        ...

    @abstractmethod
    def _inverse_transform(self, y_pred: pd.Series) -> pd.DataFrame:
        ...

    def fit(self, target_df: pd.DataFrame):
        """
        Fits the transformer to the target data.
        """
        y_true = target_df[self.target_col]
        self._fit(y_true)

    def transform(self, target_df: pd.DataFrame):
        """
        Transforms the target data using the fitted transformer.
        """
        y_true = target_df[self.target_col]
        logger.debug(f"{y_true=}")
        transformed_target_df = target_df.copy()
        transformed_target_df[self.target_col] = self._transform(y_true)
        logger.debug(f"{transformed_target_df[self.target_col]=}")
        return transformed_target_df

    def inverse_transform(self, prediction_df: pd.DataFrame):
        """
        Inverse transforms the predictions back to the original scale.
        """
        transformed_prediction_df = prediction_df.copy()
        transformed_prediction_df[self.pred_col] = self._inverse_transform(prediction_df[self.pred_col])
        return transformed_prediction_df


def process_fit_target_transformer_partition(config: Config, part: str, target_df: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    logger.info(f"Fitting target transformer on partition '{part}'...")
    target_df = get_partition(target_df, part)
    run_id = get_mlflow_run_id_for_partition(config, part)

    transformer_class = get_class_from_path(config.ds.target_transformer_class)
    transformer = transformer_class(target_col=config.mdl_task.target_col, pred_col=config.mdl_task.prediction_col)

    with mlflow.start_run(run_id=run_id, nested=True):
        transformer.fit(target_df)
        MLFlowModelLogger(transformer, "target transformer").log_model(_target_transformer_artifact_path)
        transformed_df = transformer.transform(target_df)

    return part, transformed_df

def fit_transform_target_transformer(config: Config, target_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    logger.info("Fitting target transformer for all partitions...")
    transformed_dct = {}

    with ProcessPoolExecutor(max_workers=min(len(target_df), 10)) as executor:
        futures = {
            executor.submit(process_fit_target_transformer_partition, config, part, target_df): part
            for part in target_df
        }
        for future in as_completed(futures):
            part, transformed_df = future.result()
            transformed_dct[part] = transformed_df

    return transformed_dct

def process_inverse_transform_predictions_partition_by_mlflow_model(config: Config, part: str, prediction_df: Dict[str, pd.DataFrame], mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    logger.info(f"Inverse transforming predictions on partition '{part}'...")
    preds_df = get_partition(prediction_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mlflow_run_id=mlflow_run_id)
    transformer = MLFlowModelLoader("target transformer").load_model(path=_target_transformer_artifact_path, run_id=mlflow_subrun_id)
    preds_inv_df = transformer.inverse_transform(preds_df)
    return part, preds_inv_df

def inverse_transform_predictions_by_mlflow_model(config: Config, prediction_df: Dict[str, pd.DataFrame], mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    logger.info("Inverse transforming predictions for all partitions...")
    preds_inv_dct = {}

    with ProcessPoolExecutor(max_workers=min(len(prediction_df), 10)) as executor:
        futures = {
            executor.submit(process_inverse_transform_predictions_partition_by_mlflow_model, config, part, prediction_df, mlflow_run_id): part
            for part in prediction_df
        }
        for future in as_completed(futures):
            part, preds_inv_df = future.result()
            preds_inv_dct[part] = preds_inv_df

    return preds_inv_dct
