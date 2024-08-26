import logging
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Union, List, Tuple

import mlflow
import numpy as np
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config, MetricEnum
from solvency_models.pipelines.utils.metrics import Metric
from solvency_models.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from solvency_models.pipelines.utils.utils import get_class_from_path, assert_pandas_no_lacking_indexes, \
    trunc_target_index, preds_as_dataframe_with_col_name, get_partition, get_mlflow_run_id_for_partition

logger = logging.getLogger(__name__)

# Define the artifact path for prediction model
_predictive_model_artifact_path = "model_ds/predictive_model"


class PredictiveModel(ABC):
    def __init__(self, config: Config, fit_kwargs: Dict[str, Any] = None):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.pred_col = config.mdl_task.prediction_col
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self._hparams = None
        self._features_importances = None

    def fit(self, features_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        assert_pandas_no_lacking_indexes(features_df, target_df, "features_df", "target_df")
        target_df = trunc_target_index(features_df, target_df)
        self._fit(features_df, target_df, **self._fit_kwargs, **kwargs)

    def predict(self, features_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        preds = self._predict(features_df)
        preds = preds_as_dataframe_with_col_name(features_df, preds, self.pred_col)
        return preds

    def transform(self, features_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return self.predict(features_df)

    @abstractmethod
    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _predict(self, features_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass

    def get_features_importances(self) -> pd.Series:
        return self._features_importances.copy()

    def _set_features_importances(self, features_importances: Union[pd.Series, np.ndarray]):
        if type(features_importances) is not pd.Series:
            features_importances = pd.Series(features_importances)
        features_importances.index.name = "feature"
        features_importances.name = "importance"
        self._features_importances = features_importances.sort_values(ascending=False)

    @abstractmethod
    def summary(self) -> str:
        pass

    @abstractmethod
    def get_hparams_space(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        pass

    def get_hparams(self) -> Dict[str, Any]:
        return self._hparams.copy()

    def _set_hparams(self, hparams: Dict[str, Any]):
        self._hparams = hparams


def predict_by_mlflow_model_part(config: Config, selected_features_df: pd.DataFrame,
                                 mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info("Predicting the target by the MLFlow model...")
    # Load the predictive model from the MLflow model registry
    model = MLFlowModelLoader("predictive model").load_model(path=_predictive_model_artifact_path, run_id=mlflow_run_id)
    # Make predictions
    predictions_df = model.predict(selected_features_df)
    logger.info("Made predictions.")
    return predictions_df


def fit_transform_predictive_model_part(config: Config, selected_sample_features_df: pd.DataFrame,
                                        sample_target_df: pd.DataFrame) -> pd.DataFrame:
    # Fit a model
    logger.info("Fitting the predictive model...")
    model = get_class_from_path(config.ds.model_class)(config=config)
    start_time = time.time()
    model.fit(selected_sample_features_df, sample_target_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(timedelta(seconds=elapsed_time))
    logger.info(f"Fitted the predictive model. Elapsed time: {formatted_time}.")
    # Save the model
    MLFlowModelLogger(model, "predictive model").log_model(_predictive_model_artifact_path)
    # Make predictions
    logger.info("Predicting the target...")
    predictions_df = model.predict(selected_sample_features_df)
    logger.info("Made predictions")
    return predictions_df


def join_predictions_with_target(config: Config, predictions_df: pd.DataFrame,
                                 target_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Joining the predictions with the target...")
    joined_df = target_df.join(predictions_df, how="left")
    logger.info("Joined the predictions with the target.")
    return joined_df


def evaluate_predictions_part(config: Config, predictions_df: pd.DataFrame,
                              target_df: pd.DataFrame, prefix,
                              log_to_mlflow: bool) -> List[Tuple[MetricEnum, str, float]]:
    """
    prefix: str - default None, but can by any string like train, test, pure, cal etc.
    If prefix is not None, it will be added to the metric name separated by underscore.
    E.g., train_RMSE
    """
    logger.info("Evaluating the predictions:")
    scores = []
    for metric_enum in config.mdl_task.evaluation_metrics:
        metric = Metric.from_enum(config, metric_enum, pred_col=config.mdl_task.prediction_col)
        score = metric.eval(target_df, predictions_df)
        metric_name = metric.get_short_name()
        if prefix is not None:
            metric_name = f"{prefix}_{metric_name}"
        scores.append((metric_enum, metric_name, score))
    logger.info(f"Evaluated the predictions:\n" +
                "\n".join(f"    - {name}: {score}" for _, name, score in scores))
    if log_to_mlflow:
        logger.info("Logging the metrics to MLFlow...")
        for _, metric_name, score in scores:
            mlflow.log_metric(metric_name, score)
        logger.info("Logged the metrics to MLFlow.")
    return scores


def predict_by_mlflow_model(config: Config, selected_features_df: Dict[str, pd.DataFrame],
                            mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Predicting the target by the MLFlow model...")
    for part in selected_features_df.keys():
        selected_features_part_df = get_partition(selected_features_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(part, parent_mflow_run_id=mlflow_run_id)
        logger.info(f"Predicting the target on partition '{part}' of the dataset by the MLFlow model...")
        predictions_df[part] = predict_by_mlflow_model_part(config, selected_features_part_df, mlflow_subrun_id)
    return predictions_df


def fit_transform_predictive_model(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                                   sample_target_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Fitting and transforming the predictive model...")
    for part in selected_sample_features_df.keys():
        selected_sample_features_part_df = get_partition(selected_sample_features_df, part)
        sample_target_part_df = get_partition(sample_target_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(part)
        logger.info(f"Fitting and transforming the predictive model on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            predictions_df[part] = fit_transform_predictive_model_part(config, selected_sample_features_part_df,
                                                                   sample_target_part_df)
    return predictions_df


def evaluate_predictions(config: Config, predictions_df: Dict[str, pd.DataFrame],
                         target_df: Dict[str, pd.DataFrame], prefix: str,
                         log_to_mlflow: bool) -> Dict[str, List[Tuple[MetricEnum, str, float]]]:
    scores = {}
    logger.info("Evaluating the predictions...")
    for part in predictions_df.keys():
        predictions_part_df = get_partition(predictions_df, part)
        target_part_df = get_partition(target_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(part)
        logger.info(f"Evaluating the predictions on partition '{part}' of the dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            scores[part] = evaluate_predictions_part(config, predictions_part_df, target_part_df, prefix, log_to_mlflow)
    return scores
