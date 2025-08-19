import os
import tempfile
import logging
from typing import Dict

import mlflow
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.utils.cumul_calib_curve import plot_mean_cumulative_calibration_curve, \
    plot_cumulative_calibration_curve
from claim_modelling_kedro.pipelines.utils.datasets import get_partition

logger = logging.getLogger(__name__)

FILE_NAME_MEAN_CUMUL_CALIB = "{dataset}_mean_cumul_calib.jpg"
FILE_NAME_CUMUL_CALIB = "{dataset}_cumul_calib.jpg"

ARTIFACT_PATH_CUMUL_CALIB = "summary/cumul_calib"


def get_file_name(file_template: str, dataset: str) -> str:
    """
    Returns the file name based on the template and dataset name.

    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").

    Returns:
        str: Formatted file name.
    """
    return file_template.format(dataset=dataset)


def create_mean_cumulative_calibration_curves_figs(
    config: Config,
    predictions_df: Dict[str, pd.DataFrame],
    target_df: Dict[str, pd.DataFrame],
    prediction_col: str,
    target_col: str,
    dataset: str,
    prefix: str = None,
) -> None:
    """
    Generates and logs cumulative calibration curve plots to MLflow.

    This function creates:
    1. A plot of the mean cumulative calibration curve with a standard deviation band.
    2. A plot of the mean cumulative calibration curve and the mean Lorenz curve, both with standard deviation bands.

    The resulting figures are saved as JPG files and logged to MLflow under the artifact path "summary".

    Args:
        predictions_df (Dict[str, pd.DataFrame]): Dictionary of predictions for each partition.
        target_df (Dict[str, pd.DataFrame]): Dictionary of targets for each partition.
        prediction_col (str): Name of the column containing the predictions.
        target_col (str): Name of the column containing the target values.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
    """
    y_true_dict = {part: get_partition(target_df, part)[target_col] for part in target_df.keys()}
    y_pred_dict = {part: get_partition(predictions_df, part)[prediction_col] for part in predictions_df.keys()}
    sample_weight_dict = {part: get_sample_weight(config, get_partition(target_df, part)) for part in target_df.keys()}

    # Generate the mean cumulative calibration curve
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the mean cumulative calibration curve for dataset: {dataset}...")
    fig = plot_mean_cumulative_calibration_curve(
        y_true_dict=y_true_dict,
        y_pred_dict=y_pred_dict,
        sample_weight_dict=sample_weight_dict,
        title=f"Mean Cumulative Calibration Curve ({dataset.replace('_', ' ').title()})",
        show_std_band=True,
        over_and_under_colors=True,
    )
    logger.info("Generated the mean cumulative calibration curve.")

    # Save and log the mean cumulative calibration curve to MLflow
    logger.info("Saving and logging the mean cumulative calibration curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_MEAN_CUMUL_CALIB, dataset)
        artifact_path = ARTIFACT_PATH_CUMUL_CALIB
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(f"Mean cumulative calibration curve logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")


def create_cumulative_calibration_curves_figs_part(
    config: Config,
    predictions_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    dataset: str,
    prefix: str = None,
) -> None:
    """
    Generates and logs individual cumulative calibration curve plots for a specific partition to MLflow.

    This function creates:
    1. A plot of the cumulative calibration curve.
    2. A plot of the cumulative calibration curve with the Lorenz curve.

    The resulting figures are saved as JPG files and logged to MLflow under the artifact path "summary".

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions for the partition.
        target_df (pd.DataFrame): DataFrame containing targets for the partition.
        prediction_col (str): Name of the column containing the predictions.
        target_col (str): Name of the column containing the target values.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
    """
    # Generate the cumulative calibration curve
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the cumulative calibration curve for dataset: {dataset}...")
    sample_weight = get_sample_weight(config, target_df)
    fig = plot_cumulative_calibration_curve(
        y_true=target_df[target_col],
        y_pred=predictions_df[prediction_col],
        sample_weight=sample_weight,
        title=f"Cumulative Calibration Curve ({dataset.replace('_', ' ').title()})",
        over_and_under_colors=True,
    )
    logger.info("Generated the cumulative calibration curve.")

    # Save and log the cumulative calibration curve to MLflow
    logger.info("Saving and logging the cumulative calibration curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_CUMUL_CALIB, dataset)
        artifact_path = ARTIFACT_PATH_CUMUL_CALIB
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(f"Cumulative calibration curve logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")
