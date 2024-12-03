import logging
from typing import Dict, Tuple

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.model import predict_by_mlflow_model, evaluate_predictions
from claim_modelling_kedro.pipelines.p07_data_science.select import select_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p08_model_calibration.utils import calibrate_predictions_by_mlflow_model
from claim_modelling_kedro.pipelines.p09_test.utils import remove_test_outliers
from claim_modelling_kedro.pipelines.utils.utils import get_partition, save_predictions_and_target_in_mlflow

logger = logging.getLogger(__name__)


def test_model_and_compute_stats(config: Config, calibrated_calib_predictions_df: Dict[str, pd.DataFrame],
                                 features_df: Dict[str, pd.DataFrame], target_df: Dict[str, pd.DataFrame],
                                 test_keys: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    calibrated_calib_predictions_df is only used to force the pipeline order (p09 after p08)
    """
    logger.info("Prepare test dataset...")
    test_target_df = {}
    test_features_df = {}
    for part in test_keys.keys():
        test_part_keys = get_partition(test_keys, part)
        test_target_df[part] = target_df.loc[test_part_keys, :]
        test_features_df[part] = features_df.loc[test_part_keys, :]

    # Remove outliers from the calibration dataset
    test_features_df_without_outliers, test_trg_df_without_outliers = remove_test_outliers(
        config=config,
        features_df=test_features_df,
        target_df=test_target_df
    )
    # Transform the features
    transformed_test_features_df = transform_features_by_mlflow_model(config, test_features_df_without_outliers,
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

    # Save the predictions and the target in MLFlow
    save_predictions_and_target_in_mlflow(calibrated_test_predictions_df, test_target_df, dataset="test")

    # Evaluate the predictions
    evaluate_predictions(config, pure_test_predictions_df, test_target_df, dataset="test", prefix="pure",
                         log_metrics_to_mlflow=True, save_metrics_table=True)
    evaluate_predictions(config, calibrated_test_predictions_df, test_target_df, dataset="test",
                         log_metrics_to_mlflow=True, save_metrics_table=True)

    return calibrated_test_predictions_df, test_target_df
