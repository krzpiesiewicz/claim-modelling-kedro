import logging
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Union, List, Tuple

import mlflow
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum
from claim_modelling_kedro.pipelines.utils.metrics import Metric
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path, assert_pandas_no_lacking_indexes, \
    trunc_target_index, preds_as_dataframe_with_col_name, get_partition, get_mlflow_run_id_for_partition, \
    save_metrics_table_in_mlflow, save_metrics_cv_stats_in_mlflow

logger = logging.getLogger(__name__)

# Define the artifact path for prediction model
_predictive_model_artifact_path = "model_ds/predictive_model"


def get_sample_weight(config: Config, target_df: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, None]:
    sample_weight = None
    if config.mdl_task.exposure_weighted and config.mdl_task.claim_nb_weighted:
        raise ValueError("Only one of exposure_weighted and claim_nb_weighted can be True.")
    if type(target_df) is pd.DataFrame:
        if config.mdl_task.exposure_weighted:
            sample_weight = target_df[config.data.policy_exposure_col]
            logger.debug("Using policy exposure as sample weight.")
        if config.mdl_task.claim_nb_weighted:
            sample_weight = np.fmax(target_df[config.data.claims_number_target_col], 1)
            logger.debug("Using claims number as sample weight.")
        # if sample_weight is not None:
        #     sample_weight = sample_weight / sample_weight.sum()
    return sample_weight


class PredictiveModel(ABC):
    def __init__(self, config: Config, target_col: str, pred_col: str, fit_kwargs: Dict[str, Any] = None,
                 hparams: Dict[str, Any] = None, call_updated_hparams: bool = True):
        self.config = config
        self.target_col = target_col
        self.pred_col = pred_col
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self._hparams = None
        if hparams is not None:
            self.update_hparams(hparams, call_updated_hparams=call_updated_hparams)
        self._features_importances = None
        self._not_fitted = True

    def fit(self, features_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        assert_pandas_no_lacking_indexes(features_df, target_df, "features_df", "target_df")
        target_df = trunc_target_index(features_df, target_df)
        sample_weight = get_sample_weight(self.config, target_df)
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self._fit(features_df, target_df, **self._fit_kwargs, **kwargs)
        self._not_fitted = False

    def predict(self, features_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if self._not_fitted:
            raise ValueError("The model is not fitted yet. Please, fit the model before making predictions.")
        preds = self._predict(features_df)
        logger.debug(f"{preds=}")
        preds = preds_as_dataframe_with_col_name(features_df, preds, self.pred_col)
        logger.debug(f"{preds=}")
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
        if self._not_fitted:
            raise ValueError("The model is not fitted yet. "
                             "Please, fit the model before requesting feature importances ranking.")
        if self._features_importances is None:
            raise ValueError("The model does not supply feature importances ranking.")
        return self._features_importances.copy()

    def _set_features_importances(self, features_importances: Union[pd.Series, np.ndarray]):
        if type(features_importances) is not pd.Series:
            features_importances = pd.Series(features_importances)
        features_importances.index.name = "feature"
        features_importances.name = "importance"
        self._features_importances = features_importances.sort_values(ascending=False)

    @abstractmethod
    def metric(self) -> Metric:
        pass

    @abstractmethod
    def summary(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        pass

    def get_hparams_space(self) -> Dict[str, Any]:
        return self.__class__.get_hparams_space()

    @classmethod
    @abstractmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def get_hparams_from_hyperopt_res(self, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        return self.__class__.get_hparams_from_hyperopt_res(hopt_hparams)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {}

    def get_default_hparams(self) -> Dict[str, Any]:
        return self.__class__.get_default_hparams()

    @abstractmethod
    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        pass

    def get_hparams(self) -> Dict[str, Any]:
        hparams = self.get_default_hparams()
        hparams.update(self._hparams)
        return hparams

    def update_hparams(self, hparams: Dict[str, Any] = None, call_updated_hparams: bool = True, **kwargs):
        if self._hparams is None:
            self._hparams = self.get_default_hparams().copy()
        if hparams is not None:
            self._hparams.update(hparams)
        self._hparams.update(kwargs)
        self._not_fitted = True
        if call_updated_hparams:
            self._updated_hparams()

    @abstractmethod
    def _updated_hparams(self):
        """self._hparams was updated. Now implement the logic to update the internal model with the new hyperparameters."""
        pass


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
                                        sample_target_df: pd.DataFrame, best_hparams: Dict[str, Any]) -> pd.DataFrame:
    # Fit a model
    logger.info("Fitting the predictive model...")
    model = get_class_from_path(config.ds.model_class)(config=config, target_col=config.mdl_task.target_col,
                                                       pred_col=config.mdl_task.prediction_col)
    logger.debug(f"udated hparams: {best_hparams}")
    model.update_hparams(best_hparams)
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
                              log_metrics_to_mlflow: bool) -> List[Tuple[MetricEnum, str, float]]:
    """
    prefix: str - default None, but can by any string like train, test, pure, cal etc.
    If prefix is not None, it will be added to the metric name separated by underscore.
    E.g., train_RMSE
    """
    logger.info("Evaluating the predictions:")
    scores = {}
    for metric_enum in config.mdl_task.evaluation_metrics:
        metric = Metric.from_enum(config, metric_enum, pred_col=config.mdl_task.prediction_col)
        try:
            score = metric.eval(target_df, predictions_df)
        except Exception as e:
            logger.error(f"Error while evaluating the metric {metric.get_short_name()}: {e}")
            logger.warning(f"Setting the score to NaN.")
            score = np.nan
        metric_name = metric.get_short_name()
        if prefix is not None:
            metric_name = f"{prefix}_{metric_name}"
        scores[(metric_enum, metric_name)] = score
    logger.info(f"Evaluated the predictions:\n" +
                "\n".join(f"    - {name}: {score}" for (_, name), score in scores.items()))
    if log_metrics_to_mlflow:
        logger.info("Logging the metrics to MLFlow...")
        for (_, metric_name), score in scores.items():
            mlflow.log_metric(metric_name, score)
        logger.info("Logged the metrics to MLFlow.")
    return scores


def predict_by_mlflow_model(config: Config, selected_features_df: Dict[str, pd.DataFrame],
                            mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Predicting the target by the MLFlow model...")
    for part in selected_features_df.keys():
        selected_features_part_df = get_partition(selected_features_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
        logger.info(f"Predicting the target on partition '{part}' of the dataset by the MLFlow model...")
        predictions_df[part] = predict_by_mlflow_model_part(config, selected_features_part_df, mlflow_subrun_id)
    return predictions_df


def fit_transform_predictive_model(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                                   sample_target_df: Dict[str, pd.DataFrame],
                                   best_hparams: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Fitting and transforming the predictive model...")
    for part in selected_sample_features_df.keys():
        selected_sample_features_part_df = get_partition(selected_sample_features_df, part)
        sample_target_part_df = get_partition(sample_target_df, part)
        best_hparams_part = best_hparams[part] if best_hparams is not None else None
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(f"Fitting and transforming the predictive model on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            predictions_df[part] = fit_transform_predictive_model_part(config, selected_sample_features_part_df,
                                                                       sample_target_part_df, best_hparams_part)
    return predictions_df


def evaluate_predictions(config: Config, predictions_df: Dict[str, pd.DataFrame],
                         target_df: Dict[str, pd.DataFrame], dataset: str, prefix: str = None,
                         log_metrics_to_mlflow: bool = True,
                         save_metrics_table: bool = True) -> Tuple[
    Dict[str, List[Tuple[MetricEnum, str, float]]], pd.DataFrame]:
    scores_by_part = {}
    scores_by_names = {}
    logger.info("Evaluating the predictions...")
    for part in predictions_df.keys():
        predictions_part_df = get_partition(predictions_df, part)
        target_part_df = get_partition(target_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(f"Evaluating the predictions on partition '{part}' of the dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            joined_prefix = f"{dataset}_{prefix}" if prefix is not None else dataset
            scores_by_part[part] = evaluate_predictions_part(config, predictions_part_df, target_part_df,
                                                             joined_prefix, log_metrics_to_mlflow)
        for (metric_enum, metric_name), score in scores_by_part[part].items():
            if metric_name not in scores_by_names:
                scores_by_names[metric_name] = {}
            scores_by_names[metric_name][part] = score
    # Create a DataFrame from scores_by_names
    scores_df = pd.DataFrame(scores_by_names).T
    scores_df.index.name = "metric"
    scores_df.columns.name = "part"
    if save_metrics_table:
        save_metrics_table_in_mlflow(scores_df, dataset, prefix)
    # Average the metrics over the cross-validation folds
    if config.data.cv_enabled:
        mean_df = scores_df.mean(axis=1)
        std_df = scores_df.std(axis=1)
        metrics_cv_stats_df = pd.DataFrame({"cv_mean": mean_df, "cv_std": std_df})
        if save_metrics_table:
            save_metrics_cv_stats_in_mlflow(metrics_cv_stats_df, dataset, prefix)
        if log_metrics_to_mlflow:
            for stat in metrics_cv_stats_df.columns:
                for metric_name, score in metrics_cv_stats_df[stat].items():
                    metric_name = f"{prefix}_{metric_name}" if prefix is not None else metric_name
                    mlflow.log_metric(f"{metric_name}_{stat}", score)
    elif log_metrics_to_mlflow:
        for metric_name, score in scores_df.iloc[:, 0].items():
            metric_name = f"{prefix}_{metric_name}" if prefix is not None else metric_name
            mlflow.log_metric(metric_name, score)
    return scores_by_part, scores_df
