from typing import Dict, Tuple

import mlflow
from mlflow import active_run
import logging
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p10_summary.utils.auto_calib_chart import create_auto_calib_chart_fig
from claim_modelling_kedro.pipelines.p10_summary.utils.lift_chart import create_lift_chart_fig, \
    create_lift_cv_mean_chart_fig
from claim_modelling_kedro.pipelines.p10_summary.utils.cc_lorenz import (
    create_mean_concentration_curves_figs,
    create_concentration_curves_figs_part
)
from claim_modelling_kedro.pipelines.p10_summary.utils.cumul_calib_plot import \
    create_mean_cumulative_calibration_curves_figs, create_cumulative_calibration_curves_figs_part
from claim_modelling_kedro.pipelines.p10_summary.utils.simple_lift_chart import create_simple_lift_cv_mean_chart_fig, \
    create_simple_lift_chart_fig
from claim_modelling_kedro.pipelines.p07_data_science.tabular_stats import (
    N_BINS_LIST, create_prediction_group_statistics_strict_bins
)
from claim_modelling_kedro.pipelines.utils.dataframes import load_predictions_and_target_from_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition


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

    return (sample_train_predictions_df, sample_train_target_df, sample_valid_predictions_df, sample_valid_target_df,
            calib_predictions_df, calib_target_df, train_predictions_df, train_target_df, test_predictions_df, test_target_df)


def create_curves_plots(
        config: Config,
        sample_train_predictions_df: Dict[str, pd.DataFrame],
        sample_train_target_df: Dict[str, pd.DataFrame],
        sample_valid_predictions_df: Dict[str, pd.DataFrame],
        sample_valid_target_df: Dict[str, pd.DataFrame],
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
        sample_train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample train predictions for each partition.
        sample_train_target_df (Dict[str, pd.DataFrame]): Dictionary of sample train targets for each partition.
        sample_valid_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample valid predictions for each partition.
        sample_valid_target_df (Dict[str, pd.DataFrame]): Dictionary of sample valid targets for each partition.
        calib_predictions_df (Dict[str, pd.DataFrame]): Dictionary of calibrated predictions for each partition.
        calib_target_df (Dict[str, pd.DataFrame]): Dictionary of calibrated targets for each partition.
        train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of train predictions for each partition.
        train_target_df (Dict[str, pd.DataFrame]): Dictionary of train targets for each partition.
        test_predictions_df (Dict[str, pd.DataFrame]): Dictionary of test predictions for each partition.
        test_target_df (Dict[str, pd.DataFrame]): Dictionary of test targets for each partition.
    """
    # Iterate over train and test datasets
    for dataset, predictions_df, target_df in zip(
            ["sample_train", "sample_valid", "calib", "train", "test"],
            [sample_train_predictions_df, sample_valid_predictions_df, calib_predictions_df, train_predictions_df, test_predictions_df],
            [sample_train_target_df, sample_valid_target_df, calib_target_df, train_target_df, test_target_df]):
        if len(predictions_df) == 0:
            logger.warning(f"Dataset {dataset} is empty. Skipping...")
            continue
        target_col = config.mdl_task.target_col
        prefixes_and_columns = [(None, config.mdl_task.prediction_col)]
        if config.clb.enabled and not dataset.startswith("sample"):
            prefixes_and_columns.append(("pure", config.clb.pure_prediction_col))
        for prefix, prediction_col in prefixes_and_columns:
            joined_dataset = f"{prefix}_{dataset}" if prefix is not None else dataset

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
            # Generate and log mean cumulative calibration curve
            create_mean_cumulative_calibration_curves_figs(
                config=config,
                predictions_df=predictions_df,
                target_df=target_df,
                prediction_col=prediction_col,
                target_col=target_col,
                dataset=dataset,
                prefix=prefix
            )


            # Iterate over each partition in the dataset
            for part in predictions_df.keys():
                logger.info(f"Processing partition: {part} for dataset: {joined_dataset}...")

                # Extract predictions and targets for the current partition
                part_predictions_df = get_partition(predictions_df, part)
                part_target_df = get_partition(target_df, part)

                # Get the MLflow run ID for the current partition
                mlflow_subrun_id = get_mlflow_run_id_for_partition(part, config)
                logger.info(f"Starting MLflow nested run for partition: {part} with run ID: {mlflow_subrun_id}...")

                # Start a nested MLflow run and generate individual curves
                with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
                    create_concentration_curves_figs_part(
                        config=config,
                        predictions_df=part_predictions_df,
                        target_df=part_target_df,
                        prediction_col=prediction_col,
                        target_col=target_col,
                        dataset=dataset,
                        prefix=prefix
                    )
                    create_cumulative_calibration_curves_figs_part(
                        config=config,
                        predictions_df=part_predictions_df,
                        target_df=part_target_df,
                        prediction_col=prediction_col,
                        target_col=target_col,
                        dataset=dataset,
                        prefix=prefix
                    )
    dummy_summary_1_df = pd.DataFrame({})
    return dummy_summary_1_df


def create_lift_charts(
        config: Config,
        sample_train_predictions_df: Dict[str, pd.DataFrame],
        sample_train_target_df: Dict[str, pd.DataFrame],
        sample_valid_predictions_df: Dict[str, pd.DataFrame],
        sample_valid_target_df: Dict[str, pd.DataFrame],
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
        sample_train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample train predictions for each partition.
        sample_train_target_df (Dict[str, pd.DataFrame]): Dictionary of sample train targets for each partition.
        sample_valid_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample valid predictions for each partition.
        sample_valid_target_df (Dict[str, pd.DataFrame]): Dictionary of sample valid targets for each partition.
        calib_predictions_df (Dict[str, pd.DataFrame]): Dictionary of calibrated predictions for each partition.
        calib_target_df (Dict[str, pd.DataFrame]): Dictionary of calibrated targets for each partition.
        train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of train predictions for each partition.
        train_target_df (Dict[str, pd.DataFrame]): Dictionary of train targets for each partition.
        test_predictions_df (Dict[str, pd.DataFrame]): Dictionary of test predictions for each partition.
        test_target_df (Dict[str, pd.DataFrame]): Dictionary of test targets for each partition.
    """
    # Iterate over train and test datasets
    for dataset, predictions_df, target_df in zip(
            ["sample_train", "sample_valid", "calib", "train", "test"],
            [sample_train_predictions_df, sample_valid_predictions_df, calib_predictions_df, train_predictions_df, test_predictions_df],
            [sample_train_target_df, sample_valid_target_df, calib_target_df, train_target_df, test_target_df]):
        if len(predictions_df) == 0:
            logger.warning(f"Dataset {dataset} is empty. Skipping...")
            continue
        target_col = config.mdl_task.target_col
        prefixes_and_columns = [(None, config.mdl_task.prediction_col)]
        if config.clb.enabled and not dataset.startswith("sample"):
            prefixes_and_columns.append(("pure", config.clb.pure_prediction_col))
        for prefix, prediction_col in prefixes_and_columns:
            joined_dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
            for n_bins in N_BINS_LIST:
                # Collect stats for each partition
                stats_dfs = {}
                # Iterate over each partition in the dataset
                for part in predictions_df.keys():
                    logger.info(f"Processing partition: {part} for dataset: {joined_dataset}...")

                    # Extract predictions and targets for the current partition
                    part_predictions_df = get_partition(predictions_df, part)
                    part_target_df = get_partition(target_df, part)

                    # Get the MLflow run ID for the current partition
                    mlflow_subrun_id = get_mlflow_run_id_for_partition(part, config)
                    logger.info(f"Starting MLflow nested run for partition: {part} with run ID: {mlflow_subrun_id}...")

                    # Start a nested MLflow run and generate individual concentration curves
                    with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
                        stats_df = create_prediction_group_statistics_strict_bins(
                            config=config,
                            predictions_df=part_predictions_df,
                            target_df=part_target_df,
                            prediction_col=prediction_col,
                            target_col=target_col,
                            joined_dataset=joined_dataset,
                            n_bins=n_bins,
                        )
                        logger.info(
                            f"Table of statistics for {n_bins} groups from partition: {part} in dataset: {joined_dataset} has been loaded.")
                        stats_dfs[part] = stats_df
                        create_lift_chart_fig(
                            config=config,
                            stats_df=stats_df,
                            n_bins=n_bins,
                            dataset=dataset,
                            prefix=prefix,
                        )
                        create_simple_lift_chart_fig(
                            config=config,
                            stats_df=stats_df,
                            n_bins=n_bins,
                            dataset=dataset,
                            prefix=prefix,
                        )
                        create_auto_calib_chart_fig(
                            config=config,
                            predictions_df=part_predictions_df,
                            target_df=part_target_df,
                            prediction_col=prediction_col,
                            target_col=target_col,
                            dataset=dataset,
                            n_bins=n_bins,
                            prefix=prefix
                        )

                # Average the statistics across partitions
                stats_dfs = list(stats_dfs.values())
                create_lift_cv_mean_chart_fig(
                    config=config,
                    stats_dfs=stats_dfs,
                    n_bins=n_bins,
                    dataset=dataset,
                    prefix=prefix,
                )
                create_simple_lift_cv_mean_chart_fig(
                    config=config,
                    stats_dfs=stats_dfs,
                    n_bins=n_bins,
                    dataset=dataset,
                    prefix=prefix,
                )

    dummy_summary_2_df = pd.DataFrame({})
    return dummy_summary_2_df
