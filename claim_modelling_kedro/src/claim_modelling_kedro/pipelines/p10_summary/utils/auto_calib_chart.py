import logging
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight

logger = logging.getLogger(__name__)

FILE_NAME_AUTO_CALIB_CHART = "{dataset}_auto_calib_chart_{n_bins}_groups.jpg"

ARTIFACT_PATH_AUTO_CALIB_CHARTS = "summary/auto_calib_charts"


def get_file_name(file_template: str, dataset: str, n_bins: int) -> str:
    """
    Returns the file name based on the template and dataset name.
    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the summary.
    """
    return file_template.format(dataset=dataset, n_bins=n_bins)


def plot_auto_calib_chart(
        y_true: pd.Series,
        y_pred: pd.Series,
        sample_weight: pd.Series,
        n_bins: int,
        min_x_val: float = None,
        max_x_val: float = None,
        min_y_val: float = None,
        max_y_val: float = None,
        marker: str = "o",
) -> plt.Figure:
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "weight": sample_weight or np.ones(len(y_true)),
    })

    # Sort by prediction
    df = df.sort_values("y_pred").reset_index(drop=True)

    # Assign decile bins
    df["bin"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")

    if len(np.unique(df["bin"])) == 1:
        df["bin"] = 1

    bin_means_pred = df.groupby("bin").apply(lambda g: np.average(g["y_pred"], weights=g["weight"]))
    bin_means_true = df.groupby("bin").apply(lambda g: np.average(g["y_true"], weights=g["weight"]))
    bin_bounds = df.groupby("bin")["y_pred"].agg(["min", "max"])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Histogram-style rectangles
    for i in range(len(bin_bounds)):
        xmin = bin_bounds.iloc[i]["min"]
        xmax = bin_bounds.iloc[i]["max"]
        y = bin_means_true.iloc[i]
        ax.plot([xmin, xmax], [y, y], color="black")
        ax.plot([xmin, xmin], [0, y], color="black")
        # if i < len(bin_bounds) - 1:
        ax.plot([xmax, xmax], [0, y], color="black")

    # Scatter of mean predictions vs mean true values
    ax.plot(bin_means_pred, bin_means_true, marker=marker, linestyle="", color="blue", label="Mean Target in Bin")

    # Diagonal line (perfect auto-calibration)
    low = min(min_x_val or 0, min_y_val or 0)
    high = max(max_x_val or df["y_pred"].max(), max_y_val or bin_means_true.max())
    ax.plot([low, high], [low, high], color="orange")

    # Axis limits
    if min_x_val is not None or max_x_val is not None:
        ax.set_xlim(left=min_x_val, right=max_x_val)
    if min_y_val is not None or max_y_val is not None:
        ax.set_ylim(bottom=min_y_val, top=max_y_val)

    ax.set_xlabel("Prediction")
    ax.set_title("Auto–Calibration Chart")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.close(fig)
    return fig


def create_auto_calib_chart_fig(
        config: Config,
        predictions_df: pd.DataFrame,
        target_df: pd.DataFrame,
        prediction_col: str,
        target_col: str,
        dataset: str,
        n_bins: int,
        prefix: str = None,
        min_x_val: float = None,
        max_x_val: float = None,
        min_y_val: float = None,
        max_y_val: float = None,
) -> None:
    """
    Creates a auto-chart showing the mean deviation lines between predictions and targets.

    Args:
        summary_df (pd.DataFrame): DataFrame containing summary statistics.
        n_bins (int): Number of bins used in the summary.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
        min_val (float, optional): Minimum value for y-axis. Defaults to None.
        max_val (float, optional): Maximum value for y-axis. Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the auto-calibration chart for dataset: {dataset}...")
    y_true = target_df[target_col]
    y_pred = predictions_df[prediction_col]
    sample_weight = get_sample_weight(config, target_df)
    fig = plot_auto_calib_chart(y_true, y_pred, sample_weight, n_bins=n_bins, min_x_val=min_x_val, max_x_val=max_x_val,
                                min_y_val=min_y_val, max_y_val=max_y_val)
    logger.info("Generated the auto-calibration chart.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the auto-calibration chart to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_AUTO_CALIB_CHART, dataset=dataset, n_bins=n_bins)
        artifact_path = ARTIFACT_PATH_AUTO_CALIB_CHARTS
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(
            f"Auto-calibration chart has been logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")
