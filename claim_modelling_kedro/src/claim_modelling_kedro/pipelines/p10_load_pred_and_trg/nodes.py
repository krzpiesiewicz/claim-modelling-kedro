import logging
from typing import Dict, Tuple

import pandas as pd
from mlflow import active_run

from claim_modelling_kedro.pipelines.utils.dataframes import load_predictions_and_target_from_mlflow

logger = logging.getLogger(__name__)


def load_target_and_predictions(
    dummy_test_1_df: pd.DataFrame,
    dummy_test_2_df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
           Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
           Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Loads calib, train and test predictions and targets from MLflow for the current active run.

    Args:
        dummy_test_1_df (pd.DataFrame): Placeholder DataFrame for test dataset 1.
        dummy_test_2_df (pd.DataFrame): Placeholder DataFrame for test dataset 2.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
            - Calibrated predictions and targets as a dictionary of DataFrames.
            - Train predictions and targets as a dictionary of DataFrames.
            - Test predictions and targets as a dictionary of DataFrames.
    """
    # Get the current MLflow run ID
    mlflow_run_id = active_run().info.run_id

    # Load sample train/valid predictions and targets
    sample_train_predictions_df, sample_train_target_df = load_predictions_and_target_from_mlflow(dataset="sample_train", mlflow_run_id=mlflow_run_id)
    sample_valid_predictions_df, sample_valid_target_df = load_predictions_and_target_from_mlflow(dataset="sample_valid", mlflow_run_id=mlflow_run_id)

    # Load calibrated predictions and targets
    calib_predictions_df, calib_target_df = load_predictions_and_target_from_mlflow(dataset="calib", mlflow_run_id=mlflow_run_id)

    # Load train predictions and targets
    train_predictions_df, train_target_df = load_predictions_and_target_from_mlflow(dataset="train", mlflow_run_id=mlflow_run_id)

    # Load test predictions and targets
    test_predictions_df, test_target_df = load_predictions_and_target_from_mlflow(dataset="test", mlflow_run_id=mlflow_run_id)

    dummy_load_pred_and_trg_df = pd.DataFrame({})
    return (dummy_load_pred_and_trg_df, sample_train_predictions_df, sample_train_target_df, sample_valid_predictions_df, sample_valid_target_df,
            calib_predictions_df, calib_target_df, train_predictions_df, train_target_df, test_predictions_df, test_target_df)
