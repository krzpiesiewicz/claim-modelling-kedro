import tempfile
import logging
from typing import Dict

import mlflow
import pandas as pd

from claim_modelling_kedro.pipelines.utils.concentration_curve import plot_mean_concentration_curve, \
    plot_concentration_curve
from claim_modelling_kedro.pipelines.utils.datasets import get_partition

logger = logging.getLogger(__name__)

FILE_NAME_MEAN_CC_WITH_ORACLE = "{dataset}_mean_cc_with_oracle.jpg"
FILE_NAME_MEAN_CC_AND_LORENZ = "{dataset}_mean_cc_and_lorenz.jpg"
FILE_NAME_CC_WITH_ORACLE = "{dataset}_cc_with_oracle.jpg"
FILE_NAME_CC_WITH_LORENZ = "{dataset}_cc_with_lorenz.jpg"


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


def create_mean_concentration_curves_figs(
    predictions_df: Dict[str, pd.DataFrame],
    target_df: Dict[str, pd.DataFrame],
    prediction_col: str,
    target_col: str,
    dataset: str,
    prefix: str = None,
) -> None:
    """
    Generates and logs concentration curve plots to MLflow.

    This function creates:
    1. A plot of the mean concentration curve with a standard deviation band and the oracle curve.
    2. A plot of the mean concentration curve and the mean Lorenz curve, both with standard deviation bands.

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

    # Generate the mean concentration curve with the oracle curve
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the mean concentration curve with the oracle curve for dataset: {dataset}...")
    fig1 = plot_mean_concentration_curve(
        y_true_dict=y_true_dict,
        y_pred_dict=y_pred_dict,
        title=f"Mean Concentration Curve ({dataset.replace('_', ' ').title()}) with Oracle",
        show_cc=True,
        show_oracle=True,
        show_std_band=True,
    )
    logger.info("Generated the mean concentration curve with the oracle curve.")

    # Save and log the mean concentration curve with the oracle curve to MLflow
    logger.info("Saving and logging the mean concentration curve with the oracle curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_MEAN_CC_WITH_ORACLE, dataset)
        fig1_path = f"{temp_dir}/{filename}"
        fig1.savefig(fig1_path, format="jpg")
        mlflow.log_artifact(fig1_path, artifact_path="summary")
        logger.info(f"Mean concentration curve with the oracle curve logged to MLflow under artifact path 'summary' as {filename}.")

    # Generate the mean concentration curve with the Lorenz curve
    logger.info(f"Generating the mean concentration curve with the Lorenz curve for dataset: {dataset}...")
    fig2 = plot_mean_concentration_curve(
        y_true_dict=y_true_dict,
        y_pred_dict=y_pred_dict,
        title=f"Mean Concentration Curve and Lorenz Curve ({dataset.title()}) with Oracle",
        show_cc=True,
        show_lorenz=True,
        show_std_band=True,
    )
    logger.info("Generated the mean concentration curve with the Lorenz curve.")

    # Save and log the mean concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the mean concentration curve with the Lorenz curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_MEAN_CC_AND_LORENZ, dataset)
        fig2_path = f"{temp_dir}/{filename}"
        fig2.savefig(fig2_path, format="jpg")
        mlflow.log_artifact(fig2_path, artifact_path="summary")
        logger.info(f"Mean concentration curve with the Lorenz curve logged to MLflow under artifact path 'summary' as {filename}.")


def create_concentration_curves_figs_part(
    predictions_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    dataset: str,
    prefix: str = None,
) -> None:
    """
    Generates and logs individual concentration curve plots for a specific partition to MLflow.

    This function creates:
    1. A plot of the concentration curve with the oracle curve.
    2. A plot of the concentration curve with the Lorenz curve.

    The resulting figures are saved as JPG files and logged to MLflow under the artifact path "summary".

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions for the partition.
        target_df (pd.DataFrame): DataFrame containing targets for the partition.
        prediction_col (str): Name of the column containing the predictions.
        target_col (str): Name of the column containing the target values.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
    """
    # Generate the concentration curve with the oracle curve
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the concentration curve with the oracle curve for dataset: {dataset}...")
    fig1 = plot_concentration_curve(
        y_true=target_df[target_col],
        y_pred=predictions_df[prediction_col],
        title=f"Concentration Curve ({dataset.replace('_', ' ').title()}) with Oracle",
        show_cc=True,
        show_oracle=True,
        fill_between_equity_and_cc=True,
        fill_between_cc_and_oracle=True
    )
    logger.info("Generated the concentration curve with the oracle curve.")

    # Save and log the concentration curve with the oracle curve to MLflow
    logger.info("Saving and logging the concentration curve with the oracle curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_CC_WITH_ORACLE, dataset)
        fig1_path = f"{temp_dir}/{filename}"
        fig1.savefig(fig1_path, format="jpg")
        mlflow.log_artifact(fig1_path, artifact_path="summary")
        logger.info(f"Concentration curve with the oracle curve logged to MLflow under artifact path 'summary' as {filename}.")

    # Generate the concentration curve with the Lorenz curve
    logger.info(f"Generating the concentration curve with the Lorenz curve for dataset: {dataset}...")
    fig2 = plot_concentration_curve(
        y_true=target_df[target_col],
        y_pred=predictions_df[prediction_col],
        title=f"Concentration Curve ({dataset.title()}) with Lorenz",
        show_cc=True,
        show_lorenz=True,
        fill_between_cc_and_lorenz=True
    )
    logger.info("Generated the concentration curve with the Lorenz curve.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the concentration curve with the Lorenz curve to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_CC_WITH_LORENZ, dataset)
        fig2_path = f"{temp_dir}/{filename}"
        fig2.savefig(fig2_path, format="jpg")
        mlflow.log_artifact(fig2_path, artifact_path="summary")
        logger.info(f"Concentration curve with the Lorenz curve logged to MLflow under artifact path 'summary' as {filename}.")
