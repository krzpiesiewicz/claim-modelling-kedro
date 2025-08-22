from typing import Dict, Tuple

import mlflow
import logging
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.exprmnt import Target
from claim_modelling_kedro.pipelines.p11_charts.utils.auto_calib_chart import create_auto_calib_chart_fig
from claim_modelling_kedro.pipelines.p11_charts.utils.lift_chart import create_lift_chart_fig, \
    create_lift_cv_mean_chart_fig
from claim_modelling_kedro.pipelines.p11_charts.utils.cc_lorenz import (
    create_mean_concentration_curves_figs,
    create_concentration_curves_figs_part
)
from claim_modelling_kedro.pipelines.p11_charts.utils.cumul_calib_plot import \
    create_mean_cumulative_calibration_curves_figs, create_cumulative_calibration_curves_figs_part
from claim_modelling_kedro.pipelines.p11_charts.utils.simple_lift_chart import create_simple_lift_cv_mean_chart_fig, \
    create_simple_lift_chart_fig
from claim_modelling_kedro.pipelines.p07_data_science.tabular_stats import (
    create_prediction_group_statistics_strict_bins, load_prediction_and_targets_group_stats_from_mlflow
)
from claim_modelling_kedro.pipelines.p01_init.mdl_task_config import N_BINS_LIST
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition


logger = logging.getLogger(__name__)


def create_curves_plots(
        dummy_load_pred_and_trg_df: pd.DataFrame,
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
) -> pd.DataFrame:
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
    # Iterate over sample, calib, train and test datasets
    for dataset, predictions_df, target_df in zip(
            ["sample_train", "sample_valid", "calib", "train", "test"],
            [sample_train_predictions_df, sample_valid_predictions_df, calib_predictions_df, train_predictions_df, test_predictions_df],
            [sample_train_target_df, sample_valid_target_df, calib_target_df, train_target_df, test_target_df]):
        if len(predictions_df) == 0:
            logger.warning(f"Dataset {dataset} is empty. Skipping...")
            continue
        target_col = config.mdl_task.target_col
        if dataset.startswith("sample"):
            prefixes_and_columns = [("pure", config.mdl_task.prediction_col)]
        else:
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
    dummy_charts_1_df = pd.DataFrame({})
    return dummy_charts_1_df


_MIN_AND_MAX_VALS_LIFT_CHART = {
    Target.FREQUENCY: dict(
        part = dict(
            sample = {
                10: dict(min=0, max=0.5),
                20: dict(min=0, max=0.7),
                30: dict(min=0, max=0.8),
                50: dict(min=0, max=0.9),
                100: dict(min=0, max=1.4),
            },
            full = {
                10: dict(min=0, max=0.4),
                20: dict(min=0, max=0.5),
                30: dict(min=0, max=0.55),
                50: dict(min=0, max=0.6),
                100: dict(min=0, max=0.8),
            }
        ),
        mean = dict(
            sample = {
                10: dict(min=0, max=0.4),
                20: dict(min=0, max=0.6),
                30: dict(min=0, max=0.7),
                50: dict(min=0, max=0.8),
                100: dict(min=0, max=1.1),
            },
            full = {
                10: dict(min=0, max=0.35),
                20: dict(min=0, max=0.4),
                30: dict(min=0, max=0.5),
                50: dict(min=0, max=0.5),
                100: dict(min=0, max=0.6),
            }
        )
    ),
    Target.AVG_CLAIM_AMOUNT: dict(
        part = dict(
            sample = {
                10: dict(min=None, max=None),
                20: dict(min=None, max=None),
                30: dict(min=None, max=None),
                50: dict(min=None, max=None),
                100: dict(min=None, max=None),
            },
            full = {
                10: dict(min=None, max=5000),
                20: dict(min=None, max=5000),
                30: dict(min=None, max=5000),
                50: dict(min=None, max=5000),
                100: dict(min=None, max=9500),
            }
        ),
        mean = dict(
            sample = {
                10: dict(min=None, max=None),
                20: dict(min=None, max=None),
                30: dict(min=None, max=None),
                50: dict(min=None, max=None),
                100: dict(min=None, max=None),
            },
            full = {
                10: dict(min=None, max=4000),
                20: dict(min=None, max=4000),
                30: dict(min=None, max=4000),
                50: dict(min=None, max=4000),
                100: dict(min=None, max=8500),
            }
        )
    )
}

def _get_min_max_val_lift_chart(config: Config, n_bins: int, is_sample: bool, is_part: bool) -> Tuple[float, float]:
    dct = _MIN_AND_MAX_VALS_LIFT_CHART[config.exprmnt.target]["part" if is_part else "mean"]["sample" if is_sample else "full"]
    if n_bins not in dct:
        return None, None
    return dct[n_bins]["min"], dct[n_bins]["max"]


def create_lift_charts(
        dummy_load_pred_and_trg_df: pd.DataFrame,
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
) -> pd.DataFrame:
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
    # Iterate over sample, calib, train and test datasets
    for dataset, predictions_df, target_df in zip(
            ["sample_train", "sample_valid", "calib", "train", "test"],
            [sample_train_predictions_df, sample_valid_predictions_df, calib_predictions_df, train_predictions_df, test_predictions_df],
            [sample_train_target_df, sample_valid_target_df, calib_target_df, train_target_df, test_target_df]):
        if len(predictions_df) == 0:
            logger.warning(f"Dataset {dataset} is empty. Skipping...")
            continue
        target_col = config.mdl_task.target_col
        if dataset.startswith("sample"):
            prefixes_and_columns = [("pure", config.mdl_task.prediction_col)]
        else:
            prefixes_and_columns = [(None, config.mdl_task.prediction_col)]
            if config.clb.enabled and not dataset.startswith("sample"):
                prefixes_and_columns.append(("pure", config.clb.pure_prediction_col))
        for prefix, prediction_col in prefixes_and_columns:
            joined_dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
            for n_bins in N_BINS_LIST:
                min_val, max_val = _get_min_max_val_lift_chart(config, n_bins=n_bins, is_sample=dataset.startswith("sample"), is_part=True)
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
                        stats_df = load_prediction_and_targets_group_stats_from_mlflow(
                            joined_dataset=joined_dataset,
                            n_bins=n_bins,
                            mlflow_run_id=mlflow_subrun_id,
                            raise_on_failure=False,
                        )
                        if stats_df is None:
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
                            max_val=max_val,
                            min_val=min_val,
                        )
                        create_simple_lift_chart_fig(
                            config=config,
                            stats_df=stats_df,
                            n_bins=n_bins,
                            dataset=dataset,
                            prefix=prefix,
                            max_val=max_val,
                            min_val=min_val,
                        )
                        create_auto_calib_chart_fig(
                            config=config,
                            predictions_df=part_predictions_df,
                            target_df=part_target_df,
                            prediction_col=prediction_col,
                            target_col=target_col,
                            dataset=dataset,
                            n_bins=n_bins,
                            prefix=prefix,
                            min_x_val = min_val,
                            max_x_val = max_val,
                            min_y_val = min_val,
                            max_y_val = max_val,
                        )

                # Average the statistics across partitions
                stats_dfs = list(stats_dfs.values())
                min_val, max_val = _get_min_max_val_lift_chart(config, n_bins=n_bins,
                                                               is_sample=dataset.startswith("sample"), is_part=False)
                create_lift_cv_mean_chart_fig(
                    config=config,
                    stats_dfs=stats_dfs,
                    n_bins=n_bins,
                    dataset=dataset,
                    prefix=prefix,
                    max_val=max_val,
                    min_val=min_val,
                )
                create_simple_lift_cv_mean_chart_fig(
                    config=config,
                    stats_dfs=stats_dfs,
                    n_bins=n_bins,
                    dataset=dataset,
                    prefix=prefix,
                    max_val=max_val,
                    min_val=min_val,
                )

    dummy_charts_2_df = pd.DataFrame({})
    return dummy_charts_2_df
