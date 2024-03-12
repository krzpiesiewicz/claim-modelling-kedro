import logging

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_models
from solvency_models.pipelines.p07_data_science.model import predict_by_mlflow_model, evaluate_predictions
from solvency_models.pipelines.p07_data_science.select import select_features_by_mlflow_model
from solvency_models.pipelines.p08_model_calibration.utils import calibrate_predictions_by_mlflow_model
from solvency_models.pipelines.p09_test.utils import remove_test_outliers

logger = logging.getLogger(__name__)


def test_model_and_compute_stats(config: Config, calibrated_calib_predictions_df: pd.DataFrame,
                                 features_df: pd.DataFrame, target_df: pd.DataFrame,
                                 test_keys: pd.Index) -> pd.DataFrame:
    """
    calibrated_calib_predictions_df is only used to force the pipeline order (p09 after p08)
    """
    # Get calib target
    test_target_df = target_df.loc[test_keys, :]
    test_features_df = features_df.loc[test_keys, :]
    # Remove outliers from the calibration dataset
    test_features_df_without_outliers, test_trg_df_without_outliers = remove_test_outliers(
        config=config,
        features_df=test_features_df,
        target_df=test_target_df
    )
    # Transform the features
    transformed_test_features_df = transform_features_by_mlflow_models(config, test_features_df_without_outliers,
                                                                       mlflow_run_id=config.de.mlflow_run_id)
    # Select features by the MLflow model
    selected_test_features_df = select_features_by_mlflow_model(config, transformed_test_features_df,
                                                                mlflow_run_id=config.ds.mlflow_run_id)
    # Predict by the pure MLflow model
    pure_test_predictions_df = predict_by_mlflow_model(config, selected_test_features_df,
                                                       mlflow_run_id=config.ds.mlflow_run_id)
    # Transform predictions by the MLFlow calibration model
    calibrated_test_predictions_df = calibrate_predictions_by_mlflow_model(config, pure_test_predictions_df,
                                                                           mlflow_run_id=config.clb.mlflow_run_id)
    # Evaluate the predictions
    evaluate_predictions(config, pure_test_predictions_df, test_target_df, prefix="pure_test", log_to_mlflow=True)
    evaluate_predictions(config, calibrated_test_predictions_df, test_target_df, prefix="test", log_to_mlflow=True)

    return calibrated_test_predictions_df
