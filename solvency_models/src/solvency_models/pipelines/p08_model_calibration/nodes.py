import logging

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_models
from solvency_models.pipelines.p07_data_science.model import predict_by_mlflow_model
from solvency_models.pipelines.p07_data_science.select import select_features_by_mlflow_model
from solvency_models.pipelines.p08_model_calibration.utils import remove_calib_outliers, \
    fit_transform_calibration_model, \
    calibrate_predictions_by_mlflow_model


logger = logging.getLogger(__name__)


def fit_calibration_model(config: Config, pure_sample_predictions_df: pd.DataFrame, features_df: pd.DataFrame,
                          target_df: pd.DataFrame, calib_keys) -> pd.DataFrame:
    """
    pure_sample_predictions_df is only used to force the pipeline order (p08 after p07)
    """
    if config.clb.enabled:
        # Get calib target
        calib_target_df = target_df.loc[calib_keys, :]
        calib_features_df = features_df.loc[calib_keys, :]
        # Remove outliers from the calibration dataset
        calib_features_df_without_outliers, calib_trg_df_without_outliers = remove_calib_outliers(
            config=config,
            features_df=calib_features_df,
            target_df=calib_target_df
        )
        # Transform the features
        transformed_calib_features_df = transform_features_by_mlflow_models(config, calib_features_df_without_outliers,
                                                                            mlflow_run_id=config.de.mlflow_run_id)
        # Select features by the MLflow model
        selected_calib_features_df = select_features_by_mlflow_model(config, transformed_calib_features_df,
                                                                     mlflow_run_id=config.ds.mlflow_run_id)
        # Predict by the pure MLflow model
        pure_calib_predictions_df = predict_by_mlflow_model(config, selected_calib_features_df,
                                                            mlflow_run_id=config.ds.mlflow_run_id)
        # Fit the calibration model and transform the predictions
        calibrated_calib_predictions_df = fit_transform_calibration_model(config, pure_calib_predictions_df,
                                                                          calib_trg_df_without_outliers)
        return calibrated_calib_predictions_df
    else:
        return pure_sample_predictions_df
