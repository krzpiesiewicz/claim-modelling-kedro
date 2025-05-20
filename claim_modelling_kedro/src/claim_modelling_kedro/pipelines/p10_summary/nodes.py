from typing import Dict, Tuple

import mlflow
from mlflow import active_run
import logging
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p10_summary.utils.calibration_plot import create_calibration_plot, \
    create_calibration_cv_mean_plot
from claim_modelling_kedro.pipelines.p10_summary.utils.cc_lorenz import (
    create_mean_concentration_curves_figs,
    create_concentration_curves_figs_part
)
from claim_modelling_kedro.pipelines.p10_summary.utils.tabular_stats import create_prediction_group_summary_strict_bins, \
    create_average_prediction_group_summary
from claim_modelling_kedro.pipelines.utils.dataframes import load_predictions_and_target_from_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition


logger = logging.getLogger(__name__)


def load_target_and_predictions(
    dummy_test_1_df: pd.DataFrame,
    dummy_test_2_df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
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

    # Load calibrated predictions and targets
    calib_predictions_df, calib_target_df = load_predictions_and_target_from_mlflow(dataset="calib", mlflow_run_id=mlflow_run_id)

    # Load train predictions and targets
    train_predictions_df, train_target_df = load_predictions_and_target_from_mlflow(dataset="train", mlflow_run_id=mlflow_run_id)

    # Load test predictions and targets
    test_predictions_df, test_target_df = load_predictions_and_target_from_mlflow(dataset="test", mlflow_run_id=mlflow_run_id)

    return calib_predictions_df, calib_target_df, train_predictions_df, train_target_df, test_predictions_df, test_target_df


def create_concentration_curves(
        config: Config,
        calib_predictions_df: Dict[str, pd.DataFrame],
        calib_target_df: Dict[str, pd.DataFrame],
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
    for dataset, predictions_df, target_df in zip(["calib", "train", "test"],
                                                  [calib_predictions_df, train_predictions_df, test_predictions_df],
                                                  [calib_target_df, train_target_df, test_target_df]):
        target_col = config.mdl_task.target_col
        for prefix, prediction_col in zip(["pure", None], [config.clb.pure_prediction_col, config.clb.calibrated_prediction_col]):
            dataset_name = f"{prefix}_{dataset}" if prefix is not None else dataset
            logger.info(f"Generating mean concentration curves for dataset: {dataset_name}...")

            # Generate and log mean concentration curves
            create_mean_concentration_curves_figs(
                config=config,
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
                        config=config,
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


def create_prediction_groups_stats_tables(
        config: Config,
        calib_predictions_df: Dict[str, pd.DataFrame],
        calib_target_df: Dict[str, pd.DataFrame],
        train_predictions_df: Dict[str, pd.DataFrame],
        train_target_df: Dict[str, pd.DataFrame],
        test_predictions_df: Dict[str, pd.DataFrame],
        test_target_df: Dict[str, pd.DataFrame],
) -> None:
    """
    ...

    Args:
        config (Config): Configuration object containing pipeline settings.
        train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of train predictions for each partition.
        train_target_df (Dict[str, pd.DataFrame]): Dictionary of train targets for each partition.
        test_predictions_df (Dict[str, pd.DataFrame]): Dictionary of test predictions for each partition.
        test_target_df (Dict[str, pd.DataFrame]): Dictionary of test targets for each partition.
    """
    # Iterate over train and test datasets
    for dataset, predictions_df, target_df in zip(["calib", "train", "test"],
                                                  [calib_predictions_df, train_predictions_df, test_predictions_df],
                                                  [calib_target_df, train_target_df, test_target_df]):
        target_col = config.mdl_task.target_col
        for prefix, prediction_col in zip(["pure", None], [config.clb.pure_prediction_col, config.clb.calibrated_prediction_col]):
            dataset_name = f"{prefix}_{dataset}" if prefix is not None else dataset
            for n_bins in [10, 20, 30, 50, 100]:
                # Collect stats for each partition
                summary_df = {}
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
                        stats_df = create_prediction_group_summary_strict_bins(
                            config=config,
                            predictions_df=part_predictions_df,
                            target_df=part_target_df,
                            prediction_col=prediction_col,
                            target_col=target_col,
                            dataset=dataset,
                            n_bins=n_bins,
                            prefix=prefix,
                        )
                        logger.info(
                            f"Table of statistics for {n_bins} groups from partition: {part} in dataset: {dataset_name} has been generated and logged.")
                        summary_df[part] = stats_df
                        create_calibration_plot(
                            summary_df=stats_df,
                            n_bins=n_bins,
                            dataset=dataset,
                            prefix=prefix,
                        )

                # Average the statistics across partitions
                create_average_prediction_group_summary(
                    config=config,
                    summary_df=summary_df,
                    dataset=dataset,
                    n_bins=n_bins,
                    prefix=prefix
                )
                create_calibration_cv_mean_plot(
                    summary_dfs=list(summary_df.values()),
                    n_bins=n_bins,
                    dataset=dataset,
                    prefix=prefix,
                )

    dummy_summary_2_df = pd.DataFrame({})
    return dummy_summary_2_df
