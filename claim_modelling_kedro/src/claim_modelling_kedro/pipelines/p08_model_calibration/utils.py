import logging
from typing import Tuple, Dict

import mlflow
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_models import CalibrationModel
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLogger, MLFlowModelLoader
from claim_modelling_kedro.pipelines.utils.datasets import get_class_from_path, get_mlflow_run_id_for_partition, get_partition

logger = logging.getLogger(__name__)

# Define the artifact path for calibration model
_calibration_model_artifact_path = "calibration/model"


def remove_calib_outliers_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from claim_modelling_kedro.pipelines.utils.outliers import remove_outliers as _remove_outliers

    # Remove outliers from the calibration dataset
    logger.info("Removing outliers from the calibration dataset...")
    keys_without_outliers, trg_df_without_outliers = _remove_outliers(
        target_df=target_df,
        target_col=config.mdl_task.target_col,
        lower_bound=config.clb.lower_bound,
        upper_bound=config.clb.upper_bound,
        dataset_name="calibration"
    )
    features_df_without_outliers = features_df.loc[keys_without_outliers, :]
    logger.info("Removed the outliers.")
    return features_df_without_outliers, trg_df_without_outliers


def _calibrate_by_model(config: Config, model: CalibrationModel, pure_predictions_df: pd.DataFrame) -> pd.DataFrame:
    assert config.clb.pure_prediction_col in pure_predictions_df.columns
    # Make predictions
    calibrated_predictions_df = model.predict(pure_predictions_df)
    calibrated_predictions_df = pd.concat([
        pure_predictions_df,
        calibrated_predictions_df
    ], axis=1)
    calibrated_predictions_df[config.mdl_task.prediction_col] = calibrated_predictions_df[
        config.clb.calibrated_prediction_col]
    return calibrated_predictions_df


def calibrate_predictions_by_mlflow_model_part(config: Config, pure_predictions_df: pd.DataFrame,
                                          mlflow_run_id: str) -> pd.DataFrame:
    if config.clb.enabled:
        logger.info("Calibrationg the pure predictions by the MLFlow model...")
        # Rename the prediction_col to pure_predictions_col
        pure_predictions_df = pure_predictions_df.rename(
            columns={config.mdl_task.prediction_col: config.clb.pure_prediction_col})
        # Load the predictive model from the MLflow model registry
        model = MLFlowModelLoader("calibration model").load_model(path=_calibration_model_artifact_path,
                                                                  run_id=mlflow_run_id)
        # Make predictions
        calibrated_predictions_df = _calibrate_by_model(config, model, pure_predictions_df)
        logger.info("Transformed the predictions.")
        logger.info("Calibrated predictions: %s", calibrated_predictions_df.head())
        logger.info(f"logger: {logger}")
        logger.info(f"logger.level: {logger.level}")
        return calibrated_predictions_df
    else:
        logger.info("Calibration is disabled. Hence, return the pure predictions.")
        return pure_predictions_df


def fit_transform_calibration_model_part(config: Config, pure_calib_predictions_df: pd.DataFrame,
                                    target_calib_df: pd.DataFrame) -> pd.DataFrame:
    # Fit the calibration model
    logger.info("Fitting the calibration model...")
    model = get_class_from_path(config.clb.model_class)(config=config, hparams=config.clb.model_const_hparams)
    pure_calib_predictions_df = pure_calib_predictions_df.rename(
        columns={config.mdl_task.prediction_col: config.clb.pure_prediction_col})
    model.fit(pure_calib_predictions_df, target_calib_df)
    logger.info("Fitted the calibration model.")
    # Make predictions
    logger.info("Calibrating the calib predictions using the calibration model...")
    calibrated_predictions_df = _calibrate_by_model(config, model, pure_calib_predictions_df)
    logger.info("Transformed the calib predictions.")
    # Save the model
    MLFlowModelLogger(model, "calibration model").log_model(_calibration_model_artifact_path)
    return calibrated_predictions_df


def remove_calib_outliers(
        config: Config,
        features_df: Dict[str, pd.DataFrame],
        target_df: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    features_df_without_outliers = {}
    target_df_without_outliers = {}
    logger.info("Removing outliers from the calibration dataset...")
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        target_part_df = get_partition(target_df, part)
        logger.info(f"Removing outliers from partition '{part}' of the calibration dataset...")
        features_df_without_outliers[part], target_df_without_outliers[part] = remove_calib_outliers_part(config, features_part_df, target_part_df)
    logger.info("Removed the outliers.")
    return features_df_without_outliers, target_df_without_outliers


def calibrate_predictions_by_mlflow_model(config: Config, pure_predictions_df: Dict[str, pd.DataFrame],
                                          mlflow_run_id: str) -> Dict[str, pd.DataFrame]:
    calibrated_predictions_df = {}
    logger.info("Calibrating the pure predictions by the MLFlow model...")
    for part in pure_predictions_df.keys():
        pure_predictions_part_df = get_partition(pure_predictions_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
        logger.info(f"Calibrating the pure predictions on partition '{part}' of the dataset by the MLFlow model...")
        calibrated_predictions_df[part] = calibrate_predictions_by_mlflow_model_part(config, pure_predictions_part_df, mlflow_subrun_id)
    logger.info("Calibrated the predictions.")
    return calibrated_predictions_df


def fit_transform_calibration_model(config: Config, pure_calib_predictions_df: Dict[str, pd.DataFrame],
                                    target_calib_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    calibrated_predictions_df = {}
    logger.info("Fitting and transforming the calibration model...")
    for part in pure_calib_predictions_df.keys():
        pure_calib_predictions_part_df = get_partition(pure_calib_predictions_df, part)
        target_calib_part_df = get_partition(target_calib_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(f"Fitting and transforming the calibration model on partition '{part}' of the calibration dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            calibrated_predictions_df[part] = fit_transform_calibration_model_part(config, pure_calib_predictions_part_df, target_calib_part_df)
    logger.info("Fitted and transformed the calibration model.")
    return calibrated_predictions_df
