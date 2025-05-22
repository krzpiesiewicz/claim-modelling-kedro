import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from typing import List
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


logger = logging.getLogger(__name__)


FILE_NAME_SIMPLE_LIFT_CHART = "{dataset}_simple_lift_chart_{n_bins}_groups.jpg"
FILE_NAME_SIMPLE_LIFT_MEAN_CV_CHART = "{dataset}_cv_mean_simple_lift_chart_{n_bins}_groups.jpg"

ARTIFACT_PATH_SIMPLE_LIFT_CHARTS = "summary/simple_lift_charts"


def get_file_name(file_template: str, dataset: str, n_bins: int) -> str:
    """
    Returns the file name based on the template and dataset name.
    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the summary.
    """
    return file_template.format(dataset=dataset, n_bins=n_bins)


def plot_cv_mean_simple_lift_chart(
        summary_dfs: List[pd.DataFrame],
        min_val: float = None,
        max_val: float = None,
        show_std_band: bool = True,
        marker: str = "o",
) -> plt.Figure:
    """
    Plots a minimalist lift chart with mean prediction and target lines,
    optionally showing standard deviation bands.

    Args:
        summary_dfs (List[pd.DataFrame]): List of summary DataFrames with "group", "mean_pred", and "mean_target" columns.
        min_val (float, optional): Minimum value for the y-axis. Defaults to None.
        max_val (float, optional): Maximum value for the y-axis. Defaults to None.
        show_std_band (bool, optional): Whether to show standard deviation bands. Defaults to True.
        marker (str, optional): Marker style for the chart. Defaults to "o".

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    all_groups = sorted(set(summary_dfs[0]["group"]))
    mean_pred = []
    mean_target = []
    std_pred = []
    std_target = []

    for group in all_groups:
        pred_vals = []
        target_vals = []
        for df in summary_dfs:
            row = df[df["group"] == group]
            pred_vals.append(row["mean_pred"].values[0])
            target_vals.append(row["mean_target"].values[0])
        mean_pred.append(np.mean(pred_vals))
        mean_target.append(np.mean(target_vals))
        if len(summary_dfs) > 1:
            std_pred.append(np.std(pred_vals))
            std_target.append(np.std(target_vals))

    fig, ax = plt.subplots(figsize=(8, 10))

    linewidth=2
    target_color = "black"
    pred_color = "tab:orange"


    # Plot standard deviation band if applicable
    if show_std_band and len(summary_dfs) > 1:
        ax.fill_between(all_groups, np.array(mean_target) - std_target, np.array(mean_target) + std_target,
                        color=target_color, alpha=0.1)
    # Plot mean target line
    ax.plot(all_groups, mean_target, color=target_color, marker=marker, linewidth=linewidth)

    # Plot standard deviation band if applicable
    if show_std_band and len(summary_dfs) > 1:
        ax.fill_between(all_groups, np.array(mean_pred) - std_pred, np.array(mean_pred) + std_pred,
                        color=pred_color, alpha=0.1)
    # Plot mean prediction line
    ax.plot(all_groups, mean_pred, color=pred_color, marker=marker, linewidth=linewidth)

    ax.set_xticks(all_groups)
    if len(all_groups) > 20:
        xtick_labels = [str(g) if g % 5 == 0 else "" for g in all_groups]
    else:
        xtick_labels = [str(g) for g in all_groups]
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(min(all_groups) - 0.5, max(all_groups) + 0.5)

    # Set axis limits
    if min_val is not None or max_val is not None:
        ax.set_ylim(min_val, max_val)

    # Add labels, title, and legend
    match len(all_groups):
        case 10:
            xlabel = "Decile"
        case 100:
            xlabel = "Percentile"
        case _:
            xlabel = "Group"
    ax.set_xlabel(xlabel)

    title = "Lift Chart"
    if len(summary_dfs) > 1:
        title = f"Mean {title} (CV Average)"
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color=target_color, marker=marker, linewidth=linewidth, label="Mean Target"),
        Line2D([0], [0], color=pred_color, marker=marker, linewidth=linewidth, label="Mean Prediction"),
    ]
    if show_std_band and len(summary_dfs) > 1:
        legend_handles = legend_handles + [
            Patch(facecolor=target_color, alpha=0.1, label="Standard Deviation of Target Mean"),
            Patch(facecolor=pred_color, alpha=0.1, label="Standard Deviation of Prediction Mean"),
        ]
    ax.legend(handles=legend_handles)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.close(fig)
    return fig


def create_simple_lift_chart_fig(
    summary_df: pd.DataFrame,
    n_bins: int,
    dataset: str,
    prefix: str = None,
    min_val: float = None,
    max_val: float = None) -> None:
    """
    Creates a chart showing the mean deviation lines between predictions and targets.

    Args:
        summary_df (pd.DataFrame): DataFrame containing summary statistics.
        n_bins (int): Number of bins used in the summary.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
        min_val (float, optional): Minimum value for y-axis. Defaults to None.
        max_val (float, optional): Maximum value for y-axis. Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating a simple lift chart for dataset: {dataset}...")
    fig = plot_cv_mean_simple_lift_chart([summary_df], min_val=min_val, max_val=max_val)
    logger.info("Generated the simple lift chart.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the simple lift chart to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_SIMPLE_LIFT_CHART, dataset=dataset, n_bins=n_bins)
        artifact_path = ARTIFACT_PATH_SIMPLE_LIFT_CHARTS
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(
            f"The simple lift chart has been logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")


def create_simple_lift_cv_mean_chart_fig(
    summary_dfs: List[pd.DataFrame],
    n_bins: int,
    dataset: str,
    prefix: str = None,
    min_val: float = None,
    max_val: float = None) -> None:
    """
    Creates a chart showing the mean deviation lines between predictions and targets.

    Args:
        summary_dfs (List[pd.DataFrame]): DataFrame containing summary statistics.
        n_bins (int): Number of bins used in the summary.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
        min_val (float, optional): Minimum value for y-axis. Defaults to None.
        max_val (float, optional): Maximum value for y-axis. Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the CV-mean simple lift chart for dataset: {dataset}...")
    fig = plot_cv_mean_simple_lift_chart(summary_dfs, min_val=min_val, max_val=max_val, show_std_band=True)
    logger.info("Generated the CV-mean simple lift chart.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the CV-mean simple lift chart to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_SIMPLE_LIFT_MEAN_CV_CHART, dataset=dataset, n_bins=n_bins)
        artifact_path = ARTIFACT_PATH_SIMPLE_LIFT_CHARTS
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(
            f"CV-mean simple lift chart has been logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")
