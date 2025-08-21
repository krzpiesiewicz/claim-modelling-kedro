import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Union, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.hp_config import HypertuneValidationEnum
from claim_modelling_kedro.pipelines.utils.dataframes import (
    preds_as_dataframe_with_col_name,
    save_pd_dataframe_as_csv_in_mlflow
)
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.metrics import Metric
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight

logger = logging.getLogger(__name__)

# Define the artifact path for prediction model
_ds_artifact_path = "model_ds"
_model_name = "predictive model"
_predictive_model_artifact_path = f"{_ds_artifact_path}/{_model_name}"
_features_importance_filename = "model_features_importance.csv"


class PredictiveModel(ABC):
    def __init__(self, config: Config, target_col: str, pred_col: str, fit_kwargs: Dict[str, Any] = None,
                 hparams: Dict[str, Any] = None, call_updated_hparams: bool = True):
        self.config = config
        self.target_col = target_col
        self.pred_col = pred_col
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self._hparams = None
        self.update_hparams(hparams, call_updated_hparams=call_updated_hparams)
        self._features_importance = None
        self._not_fitted = True

    def fit(self,
            features_df: pd.DataFrame,
            target_df: pd.DataFrame,
            sample_train_keys: Optional[pd.Index] = None,
            sample_val_keys: Optional[pd.Index] = None,
            **kwargs):
        data_index = features_df.index.intersection(target_df.index)
        if sample_train_keys is not None:
            sample_train_keys = sample_train_keys.intersection(data_index)
        else:
            sample_train_keys = data_index.difference(sample_val_keys if sample_val_keys is not None else pd.Index([]))
        if sample_val_keys is not None:
            sample_val_keys = sample_val_keys.intersection(data_index)
        data_index = data_index.intersection(sample_train_keys.union(sample_val_keys if sample_val_keys is not None else pd.Index([])))
        features_df = features_df.loc[data_index, :]
        target_df = target_df.loc[data_index, :]
        if not "sample_weight" in kwargs:
            sample_weight = get_sample_weight(self.config, target_df)
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
        if "sample_weight" in kwargs:
            kwargs["sample_weight"] = kwargs["sample_weight"].loc[data_index]
        self._fit(features_df, target_df, sample_train_keys=sample_train_keys, sample_val_keys=sample_val_keys,
                  **self._fit_kwargs, **kwargs)
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
    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, sample_train_keys: pd.Index,
             sample_val_keys: Optional[pd.Index] = None, **kwargs):
        pass

    @abstractmethod
    def _predict(self, features_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass

    def get_features_importance(self) -> pd.Series:
        if self._not_fitted:
            raise ValueError("The model is not fitted yet. "
                             "Please, fit the model before requesting feature importance ranking.")
        if self._features_importance is None:
            raise ValueError("The model does not supply feature importance ranking.")
        return self._features_importance.copy()

    def _set_features_importance(self, features_importance: Union[pd.Series, np.ndarray]):
        if type(features_importance) is not pd.Series:
            features_importance = pd.Series(features_importance)
        features_importance.index.name = "feature"
        features_importance.name = "importance"
        self._features_importance = features_importance.sort_values(ascending=False)

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
        return hparams.copy()

    def update_hparams(self, hparams: Dict[str, Any] = None, call_updated_hparams: bool = True, **kwargs):
        _current_hparams = self._hparams or {}
        self._hparams = self.get_default_hparams().copy()
        self._hparams.update(_current_hparams)
        if hparams is not None:
            self._hparams.update({hparam: value for hparam, value in hparams.items() if value is not None})
        self._hparams.update(kwargs)
        self._not_fitted = True
        if call_updated_hparams:
            self._updated_hparams()

    @abstractmethod
    def _updated_hparams(self):
        """self._hparams was updated. Now implement the logic to update the internal model with the new hyperparameters."""
        pass


def save_ds_model_in_mlflow(model: PredictiveModel):
    MLFlowModelLogger(model, _model_name).log_model(_predictive_model_artifact_path)


def load_ds_model_from_mlflow(mlflow_run_id: str = None) -> PredictiveModel:
    return MLFlowModelLoader(_model_name).load_model(path=_predictive_model_artifact_path, run_id=mlflow_run_id)


def predict_by_mlflow_model_part(config: Config, selected_features_df: pd.DataFrame, part: str,
                                 mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info(f"Predicting the target by the MLFlow model for partition '{part}'...")
    # Load the predictive model from the MLflow model registry
    model = load_ds_model_from_mlflow(mlflow_run_id=mlflow_run_id)
    # Make predictions
    predictions_df = model.predict(selected_features_df)
    logger.info(f"Made predictions for partition '{part}'.")
    return predictions_df


def fit_transform_predictive_model_part(config: Config, selected_sample_features_df: pd.DataFrame,
                                        sample_target_df: pd.DataFrame, sample_train_keys: pd.Index,
                                        sample_val_keys: pd.Index, best_hparams: Dict[str, Any], part: str,
                                        mlflow_run_id: str) -> pd.DataFrame:
    if config.ds.hp.enabled and config.ds.hp.validation_method == HypertuneValidationEnum.SAMPLE_VAL_SET:
        # use the best model from the hypertuning
        logger.info(f"Using the best model from hypertuning for partition '{part}' - loading the model from mlflow...")
        model = load_ds_model_from_mlflow(mlflow_run_id=mlflow_run_id)
    else:
        # use the best hyperparameters from the hypertuning to fit a new model
        assert best_hparams is not None, "best_hparams should be provided when using hypertuning validation method other than SAMPLE_VAL_SET"
        logger.info(f"Fitting the predictive model for partition '{part}'...")
        model = get_class_from_path(config.ds.model_class)(config=config, target_col=config.mdl_task.target_col,
                                                           pred_col=config.mdl_task.prediction_col)
        model.update_hparams(config.ds.model_const_hparams)
        logger.debug(f"udated hparams for partition '{part}': {best_hparams}")
        model.update_hparams(best_hparams)
        logger.info(f" for partition '{part}': {model.get_hparams()=}")
        start_time = time.time()
        model.fit(selected_sample_features_df, sample_target_df, sample_train_keys=sample_train_keys, sample_val_keys=sample_val_keys)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logger.info(f"Fitted the predictive model for partition '{part}'. Elapsed time: {formatted_time}.")
        save_ds_model_in_mlflow(model)
    # Save the features importance
    logger.info(f"Saving the features importance for partition '{part}'...")
    try:
        features_importance = model.get_features_importance().reset_index()
        save_pd_dataframe_as_csv_in_mlflow(features_importance, _ds_artifact_path, _features_importance_filename,
                                           index=True)
        logger.info(
            f"Successfully saved the features importance for partition '{part}' to MLFlow path {os.path.join(_ds_artifact_path, _features_importance_filename)}")
    except ValueError as e:
        logger.warning(f"{e}. Skipping saving features importance for partition '{part}'.")
    # Make predictions
    logger.info(f"Predicting the target for partition '{part}'...")
    predictions_df = model.predict(selected_sample_features_df)
    logger.info(f"Made predictions for partition '{part}'")
    return predictions_df


def process_predict_partition(config: Config, part: str, selected_features_df: Dict[str, pd.DataFrame],
                              mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    selected_features_part_df = get_partition(selected_features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(part, config, parent_mlflow_run_id=mlflow_run_id)
    predictions_part_df = predict_by_mlflow_model_part(config, selected_features_part_df, part, mlflow_subrun_id)
    return part, predictions_part_df


def predict_by_mlflow_model(config: Config, selected_features_df: Dict[str, pd.DataFrame],
                            mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Predicting the target by the MLFlow model...")

    parts_cnt = len(selected_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_predict_partition, config, part, selected_features_df, mlflow_run_id): part
            for part in selected_features_df.keys()
        }
        for future in as_completed(futures):
            part, predictions_part_df = future.result()
            predictions_df[part] = predictions_part_df

    return predictions_df


from concurrent.futures import ProcessPoolExecutor, as_completed


def process_part(config: Config, part: str, selected_sample_features_df: Dict[str, pd.DataFrame],
                 sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                 sample_val_keys: Dict[str, pd.Index], best_hparams: Dict[str, Dict[str, Any]]) -> Tuple[
    str, pd.DataFrame]:
    selected_sample_features_part_df = get_partition(selected_sample_features_df, part)
    sample_target_part_df = get_partition(sample_target_df, part)
    sample_train_keys_part = get_partition(sample_train_keys, part)
    sample_val_keys_part = get_partition(sample_val_keys, part)
    best_hparams_part = best_hparams[part] if best_hparams is not None else None
    mlflow_subrun_id = get_mlflow_run_id_for_partition(part, config)
    logger.info(f"Fitting and transforming the predictive model on partition '{part}' of the sample dataset...")
    with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
        return part, fit_transform_predictive_model_part(config, selected_sample_features_part_df,
                                                         sample_target_part_df, sample_train_keys_part,
                                                         sample_val_keys_part, best_hparams_part, part,
                                                         mlflow_subrun_id)


def fit_transform_predictive_model(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                                   sample_target_df: Dict[str, pd.DataFrame],
                                   sample_train_keys: Dict[str, pd.Index], sample_val_keys: Dict[str, pd.Index],
                                   best_hparams: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    predictions_df = {}
    logger.info("Fitting and transforming the predictive model...")

    parts_cnt = len(selected_sample_features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {executor.submit(process_part, config, part, selected_sample_features_df, sample_target_df,
                                   sample_train_keys, sample_val_keys, best_hparams): part
                   for part in selected_sample_features_df.keys()}
        for future in as_completed(futures):
            part, result = future.result()
            predictions_df[part] = result

    return predictions_df


