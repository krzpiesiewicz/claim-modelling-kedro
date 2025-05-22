import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from typing import List
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


FILE_NAME_LIFT_CHART = "{dataset}_lift_chart_{n_bins}_groups.jpg"
FILE_NAME_LIFT_MEAN_CV_CHART = "{dataset}_cv_mean_lift_chart_{n_bins}_groups.jpg"

ARTIFACT_PATH_LIFT_CHARTS = "summary/lift_charts"


def get_file_name(file_template: str, dataset: str, n_bins: int) -> str:
    """
    Returns the file name based on the template and dataset name.
    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the summary.
    """
    return file_template.format(dataset=dataset, n_bins=n_bins)


def plot_cv_mean_lift_chart(
        summary_dfs: List[pd.DataFrame],
        min_val: float = None,
        max_val: float = None,
        interpolate_pred_mean: bool = True
) -> plt.Figure:
    all_groups = sorted(set(summary_dfs[0]["group"]))
    mean_pred = []
    mean_target = []
    std_pred = []
    red_heights = []
    blue_heights = []

    vline_width = np.interp([len(all_groups)], [1, 3, 7, 10, 20, 100], [100, 70, 40, 30, 10, 3])[0]

    for group in all_groups:
        pred_vals, target_vals, red_parts, blue_parts = [], [], [], []
        for df in summary_dfs:
            row = df[df["group"] == group]
            pred = row["mean_pred"].values[0]
            target = row["mean_target"].values[0]
            pred_vals.append(pred)
            target_vals.append(target)
            red_parts.append(max(0, pred - target))
            blue_parts.append(max(0, target - pred))
            # if pred >= target:
            #     red_parts.append(pred - target)
            # if pred <= target:
            #     blue_parts.append(target - pred)
        mean_pred.append(np.mean(pred_vals))
        mean_target.append(np.mean(target_vals))
        std_pred.append(np.std(pred_vals))
        red_heights.append(np.mean(red_parts) if len(red_parts) > 0 else 0)
        blue_heights.append(np.mean(blue_parts) if len(blue_parts) > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 10))

    # Mean prediction
    if interpolate_pred_mean:
        ax.plot(all_groups, mean_pred, color="black", linewidth=3)
    else:
        ax.plot(all_groups, mean_pred, color="gray", linewidth=2)
        for i, group in enumerate(all_groups):
            ax.hlines(y=mean_pred[i], xmin=group - 0.4, xmax=group + 0.4, color="black", linewidth=4, label=None)

    # Standard deviation band
    if len(summary_dfs) > 1:
        ax.fill_between(all_groups, np.array(mean_pred) - std_pred, np.array(mean_pred) + std_pred,
                        color="black", alpha=0.1)

    # Over/under bars
    for i, group in enumerate(all_groups):
        base = mean_pred[i]
        if red_heights[i] > 0:
            ax.vlines(group, ymin=base - red_heights[i], ymax=base, color="red", linewidth=vline_width, alpha=0.5)
        if blue_heights[i] > 0:
            ax.vlines(group, ymin=base, ymax=base + blue_heights[i], color="blue", linewidth=vline_width, alpha=0.5)

        # Mean target as a horizontal line
        target_y = mean_target[i]
        if target_y > base:
            marker_color = "blue"
        elif target_y < base:
            marker_color = "red"
        else:
            marker_color = "black"
        ax.hlines(y=target_y, xmin=group - 0.4, xmax=group + 0.4, color=marker_color, linewidth=4)

    # X-axis: grid every 1, labels every 5 (if many groups)
    ax.set_xticks(all_groups)
    if len(all_groups) > 25:
        xtick_labels = [str(g) if g % 5 == 0 or g == 1 else "" for g in all_groups]
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
    title = "Extended Lift Chart"
    if len(summary_dfs) > 1:
        title = f"{title} (CV Average)"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

    label_dev_template = "Mean Bias Deviation of Target Mean ({kind} Part)" if len(
        summary_dfs) > 1 else "Bias Deviation of Target Mean (When {kind})"
    # Legend
    legend_handles = []
    if not interpolate_pred_mean:
        legend_handles.append(
            Line2D([0], [0], color='black', marker='_', markersize=17, markeredgewidth=4, linestyle='None',
                   label="Mean Prediction"))
    else:
        legend_handles.append(Line2D([0], [0], color="black", linewidth=4, label="Mean Prediction"))
    if len(summary_dfs) > 1:
        legend_handles.append(Line2D([0], [0], color="black", linewidth=min(vline_width, 10), alpha=0.1,
                                     label="Standard Deviation of Mean Prediction")),

    legend_handles = legend_handles + [
        Line2D([0], [0], color='blue', marker='_', markersize=17, markeredgewidth=4, linestyle='None',
               label="Mean Target (When Underpriced)"),
        Line2D([0], [0], color='red', marker='_', markersize=17, markeredgewidth=4, linestyle='None',
               label="Mean Target (When Overpriced)"),
        Line2D([0], [0], color="blue", linewidth=min(vline_width, 10), alpha=0.5,
               label=label_dev_template.format(kind="Underpriced")),
        Line2D([0], [0], color="red", linewidth=min(vline_width, 10), alpha=0.5,
               label=label_dev_template.format(kind="Overpriced")),
    ]

    ax.legend(handles=legend_handles)
    plt.close(fig)
    return fig


def create_lift_chart_fig(
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
        min_val (float, optional): Minimum value for y-axis. Defaults to None.
        max_val (float, optional): Maximum value for y-axis. Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the lift chart for dataset: {dataset}...")
    fig = plot_cv_mean_lift_chart([summary_df], min_val, max_val, interpolate_pred_mean=False)
    logger.info("Generated the lift chart.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the lift chart to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_LIFT_CHART, dataset=dataset, n_bins=n_bins)
        artifact_path = ARTIFACT_PATH_LIFT_CHARTS
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(
            f"lift chart has been logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")


def create_lift_cv_mean_chart_fig(
    summary_dfs: List[pd.DataFrame],
    n_bins: int,
    dataset: str,
    prefix: str = None,
    min_val: float = None,
    max_val: float = None) -> None:
    """
    Creates a chart showing the mean deviation lines between predictions and targets.

    Args:
        summary_df (pd.DataFrame): DataFrame containing summary statistics.
        min_val (float, optional): Minimum value for y-axis. Defaults to None.
        max_val (float, optional): Maximum value for y-axis. Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating the CV-mean lift chart for dataset: {dataset}...")
    fig = plot_cv_mean_lift_chart(summary_dfs, min_val, max_val, interpolate_pred_mean=False)
    logger.info("Generated the CV-mean lift chart.")

    # Save and log the concentration curve with the Lorenz curve to MLflow
    logger.info("Saving and logging the CV-mean lift chart to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_LIFT_MEAN_CV_CHART, dataset=dataset, n_bins=n_bins)
        artifact_path = ARTIFACT_PATH_LIFT_CHARTS
        fig_path = os.path.join(temp_dir, filename)
        fig.savefig(fig_path, format="jpg")
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        logger.info(
            f"CV-mean lift chart has been logged to MLflow under artifact {os.path.join(artifact_path, filename)}.")