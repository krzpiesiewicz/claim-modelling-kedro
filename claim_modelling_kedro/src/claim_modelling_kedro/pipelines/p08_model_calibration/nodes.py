import logging
from typing import Dict

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.model import predict_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.pred_eval import evaluate_predictions
from claim_modelling_kedro.pipelines.p07_data_science.select import select_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import inverse_transform_predictions_by_mlflow_model
from claim_modelling_kedro.pipelines.p08_model_calibration.utils import handle_outliers, \
    fit_transform_calibration_model
from claim_modelling_kedro.pipelines.utils.datasets import get_partition
from claim_modelling_kedro.pipelines.utils.dataframes import save_predictions_and_target_in_mlflow

logger = logging.getLogger(__name__)


def fit_calibration_model(config: Config, pure_sample_predictions_df: Dict[str, pd.DataFrame],
                          features_df: pd.DataFrame, target_df: pd.DataFrame,
                          calib_keys: Dict[str, pd.Index]) -> pd.DataFrame:
    """
    pure_sample_predictions_df is only used to force the pipeline order (p08 after p07)
    """
    if config.clb.enabled:
        calib_target_df = {}
        calib_features_df = {}
        logger.info("Prepare calibration dataset...")
        for part in calib_keys.keys():
            calib_part_keys = get_partition(calib_keys, part)
            calib_target_df[part] = target_df.loc[calib_part_keys, :]
            calib_features_df[part] = features_df.loc[calib_part_keys, :]

        # Handle outliers in the calibration dataset
        calib_features_df_handled_outliers, calib_trg_df_handled_outliers = handle_outliers(
            config=config,
            features_df=calib_features_df,
            target_df=calib_target_df
        )
        # Transform the features
        transformed_calib_features_df = transform_features_by_mlflow_model(config,
                                                                           calib_features_df_handled_outliers,
                                                                           mlflow_run_id=config.de.mlflow_run_id)
        # Select features by the MLflow model
        if config.ds.fs.enabled:
            selected_calib_features_df = select_features_by_mlflow_model(config, transformed_calib_features_df,
                                                                         mlflow_run_id=config.ds.mlflow_run_id)
        else:
            selected_calib_features_df = transformed_calib_features_df
        # Predict by the pure MLflow model
        pure_calib_predictions_df = predict_by_mlflow_model(config, selected_calib_features_df,
                                                            mlflow_run_id=config.ds.mlflow_run_id)
        if config.ds.trg_trans.enabled:
            # Inverse transform the predictions
            pure_calib_predictions_df = inverse_transform_predictions_by_mlflow_model(config, pure_calib_predictions_df,
                                                                                      mlflow_run_id=config.ds.mlflow_run_id)

        # Fit the calibration model and transform the predictions
        calibrated_calib_predictions_df = fit_transform_calibration_model(config, pure_calib_predictions_df,
                                                                          calib_trg_df_handled_outliers)
        # Save the predictions and the target in MLFlow
        save_predictions_and_target_in_mlflow(calibrated_calib_predictions_df, calib_target_df, dataset="calib")

        # Evaluate the predictions
        evaluate_predictions(config, pure_calib_predictions_df, calib_target_df, dataset="calib", prefix="pure",
                             log_metrics_to_mlflow=True, save_metrics_table=True)
        evaluate_predictions(config, calibrated_calib_predictions_df, calib_target_df, dataset="calib",
                             log_metrics_to_mlflow=True, save_metrics_table=True)
        return calibrated_calib_predictions_df
    else:
        return pure_sample_predictions_df
