from typing import Dict, Tuple

import mlflow
from mlflow import active_run
import logging
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p10_summary.uitls import (
    create_mean_concentration_curves_figs,
    create_concentration_curves_figs_part
)
from claim_modelling_kedro.pipelines.utils.dataframes import load_predictions_and_target_from_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition


logger = logging.getLogger(__name__)


def load_train_and_test_predictions(
    dummy_test_1_df: pd.DataFrame,
    dummy_test_2_df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Loads train and test predictions and targets from MLflow for the current active run.

    Args:
        dummy_test_1_df (pd.DataFrame): Placeholder DataFrame for test dataset 1.
        dummy_test_2_df (pd.DataFrame): Placeholder DataFrame for test dataset 2.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
            - Train predictions and targets as a dictionary of DataFrames.
            - Test predictions and targets as a dictionary of DataFrames.
    """
    # Get the current MLflow run ID
    mlflow_run_id = active_run().info.run_id

    # Load train predictions and targets
    train_predictions_df, train_target_df = load_predictions_and_target_from_mlflow(dataset="train", mlflow_run_id=mlflow_run_id)

    # Load test predictions and targets
    test_predictions_df, test_target_df = load_predictions_and_target_from_mlflow(dataset="test", mlflow_run_id=mlflow_run_id)

    return train_predictions_df, train_target_df, test_predictions_df, test_target_df


def create_concentration_curves(
        config: Config,
        train_predictions_df: Dict[str, pd.DataFrame],
        train_target_df: Dict[str, pd.DataFrame],
        test_predictions_df: Dict[str, pd.DataFrame],
        test_target_df: Dict[str, pd.DataFrame],
) -> None:
    """
    Generates and logs concentration curve plots to MLflow.

    This function creates:
    1. A plot of the mean concentration curve with a standard deviation band and the oracle curve.
    2. A plot of the mean concentration curve and the mean Lorenz curve, both with standard deviation bands.

    The resulting figures are saved as JPG files and logged to MLflow under the artifact path "summary".

    Args:
        config (Config): Configuration object containing pipeline settings.
        train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of train predictions for each partition.
        train_target_df (Dict[str, pd.DataFrame]): Dictionary of train targets for each partition.
        test_predictions_df (Dict[str, pd.DataFrame]): Dictionary of test predictions for each partition.
        test_target_df (Dict[str, pd.DataFrame]): Dictionary of test targets for each partition.
    """
    # Iterate over train and test datasets
    for dataset, predictions_df, target_df in zip(["train", "test"], [train_predictions_df, test_predictions_df],
                                                  [train_target_df, test_target_df]):
        target_col = config.mdl_task.target_col
        for prefix, prediction_col in zip(["pure", None], [config.clb.pure_prediction_col, config.clb.calibrated_prediction_col]):
            dataset_name = f"{prefix}_{dataset}" if prefix is not None else dataset
            logger.info(f"Generating mean concentration curves for dataset: {dataset_name}...")

            # Generate and log mean concentration curves
            create_mean_concentration_curves_figs(
                predictions_df=predictions_df,
                target_df=target_df,
                prediction_col=prediction_col,
                target_col=target_col,
                dataset=dataset,
                prefix=prefix
            )
            logger.info(f"Mean concentration curves for dataset: {dataset_name} have been generated and logged.")

            # Iterate over each partition in the dataset
            for part in predictions_df.keys():
                logger.info(f"Processing partition: {part} for dataset: {dataset_name}...")

                # Extract predictions and targets for the current partition
                part_predictions_df = get_partition(predictions_df, part)
                part_target_df = get_partition(target_df, part)

                # Get the MLflow run ID for the current partition
                mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
                logger.info(f"Starting MLflow nested run for partition: {part} with run ID: {mlflow_subrun_id}...")

                # Start a nested MLflow run and generate individual concentration curves
                with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
                    create_concentration_curves_figs_part(
                        predictions_df=part_predictions_df,
                        target_df=part_target_df,
                        prediction_col=prediction_col,
                        target_col=target_col,
                        dataset = dataset,
                        prefix=prefix
                    )
                logger.info(
                    f"Concentration curves for partition: {part} in dataset: {dataset_name} have been generated and logged.")
    dummy_summary_1_df = pd.DataFrame({})
    return dummy_summary_1_df
