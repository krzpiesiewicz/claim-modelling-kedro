from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index


logger = logging.getLogger(__name__)


def calculate_cumulative_calibration_curve(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
    """
    Calculate the points for the Cumulative Calibration Curve.

    Args:
        y_true (pd.Series or pd.Series):
        y_pred (pd.Series or pd.Series):
        sample_weight (pd.Series or pd.Series, optional): Observation weights. Defaults to None.

    Returns:
        tuple: (x_points, y_points) for the Concentration Curve.
    """
    y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)

    # Weighted cumulative sums
    weighted_y_true = y_true * sample_weight
    weighted_y_pred = y_pred * sample_weight

    # Total sums for normalization
    total_y_true = weighted_y_true.sum()
    total_y_pred = weighted_y_pred.sum()

    # Cumulative sums
    cum_y_true = np.cumsum(weighted_y_true) / total_y_true
    cum_y_pred = np.cumsum(weighted_y_pred) / total_y_pred

    # Include the origin point (0, 0)
    cum_y_true = np.insert(cum_y_true, 0, 0)
    cum_y_pred = np.insert(cum_y_pred, 0, 0)

    return cum_y_pred, cum_y_true


def segment_intersects_y_eq_x(p0, p1):
    """
    Check if a line segment intersects the line y = x.

    Args:
        p0 (tuple): Starting point of the segment (x0, y0).
        p1 (tuple): Ending point of the segment (x1, y1).

    Returns:
        tuple or None: Intersection point (xi, yi) if the segment intersects y = x, otherwise None.
    """
    x0, y0 = p0
    x1, y1 = p1

    # Check if the segment crosses the line y = x (sign change check)
    f0 = y0 - x0
    f1 = y1 - x1

    if f0 * f1 >= 0:
        return None  # No intersection on the open segment

    # Parametric equation of the segment: x = x0 + t*(x1 - x0), y = y0 + t*(y1 - y0)
    # Solve for t where x = y -> x0 + t*(x1 - x0) = y0 + t*(y1 - y0)
    numerator = y0 - x0
    denominator = (x1 - x0) - (y1 - y0)
    if denominator == 0:
        return None  # Parallel to y = x or collinear
    t = numerator / denominator

    if 0 < t < 1:  # Only consider points inside the segment (not endpoints)
        xi = x0 + t * (x1 - x0)
        yi = y0 + t * (y1 - y0)
        return (xi, yi)
    else:
        return None


def plot_cumulative_calibration_curve(
        y_true: pd.Series,
        y_pred: pd.Series,
        sample_weight: pd.Series = None,
        title: str = "Cumulative Calibration Curve",
        n_points: int = 100,
        fill_between_equity_and_curve: bool = True,
        over_and_under_colors: bool = True
) -> plt.Figure:
    """
    Plot the Cumulative Calibration Curve with optional overpricing and underpricing coloring.

    Args:
        y_true (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
        sample_weight (Optional[pd.Series]): Weights for observations (optional).
        title (str): Title of the plot.
        n_points (int): Number of interpolation points for the curves.
        fill_between_equity_and_curve (bool): Whether to fill the area between the curve and the line of equality.
        over_and_under_colors (bool): Whether to color overpricing and underpricing parts differently.

    Returns:
        Figure: Matplotlib figure object.
    """
    x_interp = np.linspace(0, 1, n_points)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate the cumulative calibration curve
    x, y = calculate_cumulative_calibration_curve(y_true, y_pred, sample_weight)
    y_interp = np.interp(x_interp, x, y)

    if over_and_under_colors:
        overpricing_parts = []
        underpricing_parts = []
        stack = [(x_interp[0], y_interp[0])]
        for i in range(len(x_interp) - 1):
            p0 = x_interp[i], y_interp[i]
            p1 = x_interp[i + 1], y_interp[i + 1]
            intersection = segment_intersects_y_eq_x(p0, p1)
            lays_on_eq = (p0[0] == p0[1] and p1[0] == p1[1])
            if lays_on_eq:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="black")
                stack = []
            elif intersection is not None or p1[0] == p1[1] or i == len(x_interp) - 2:
                stack.append(intersection or p1)
                if p0[1] > p0[0]:  # Above the equality line
                    overpricing_parts.append(stack)
                else:  # Below the equality line
                    underpricing_parts.append(stack)
                stack = [intersection] if intersection is not None else []
            stack.append(p1)

        # Plot overpricing and underpricing parts
        for parts, color in zip([overpricing_parts, underpricing_parts], ["blue", "red"]):
            for part in parts:
                x = list(map(lambda p: p[0], part))
                y = list(map(lambda p: p[1], part))
                if fill_between_equity_and_curve:
                    ax.fill_between(x, y, x, color=color, alpha=0.05)
                ax.plot(x, y, color=color)
        legend_handles = [
            Line2D([0], [0], color="blue", label="Cumulative Calibration Curve (Underpriced Part)"),
            Line2D([0], [0], color="red", label="Cumulative Calibration Curve (Overpriced Part)"),
        ]
    else:
        # Plot the curve without over/under coloring
        if fill_between_equity_and_curve:
            ax.fill_between(x_interp, y_interp, y2=x_interp, color="green", alpha=0.05)
        ax.plot(x_interp, y_interp, label="Cumulative Calibration Curve", color="green")
        legend_handles = [
            Line2D([0], [0], color="green", label="Cumulative Calibration Curve (Underpriced Part)"),
        ]

    # Plot the line of equality
    ax.plot([0, 1], [0, 1], label="Line of Equality", linestyle="dotted", color="black")
    legend_handles.append(Line2D([0], [0], color="black", linestyle="dotted", label="Line of Equality"))

    # Set axis labels, title, and legend
    ax.set_xlabel("Cumulative Proportion of predicted severity")
    ax.set_ylabel("Cumulative Proportion of claims sizes")
    ax.set_title(title)
    ax.legend(handles=legend_handles)
    ax.grid(True)
    plt.close(fig)
    return fig


def plot_mean_cumulative_calibration_curve(
    y_true_dict: Dict[str, pd.Series],
    y_pred_dict: Dict[str, pd.Series],
    sample_weight_dict: Optional[Dict[str, pd.Series]] = None,
    title: str = "Mean Cumulative Calibration Curve across Partitions",
    fill_between_equity_and_curve: bool = False,
    over_and_under_colors: bool = False,
    show_std_band: bool = True,
    show_all_curves: bool = False,
    n_points: int = 100
) -> plt.Figure:
    x_interp = np.linspace(0, 1, n_points)
    y_interp_all = []

    fig, ax = plt.subplots(figsize=(8, 6))

    for part in y_true_dict:
        weights = sample_weight_dict.get(part) if sample_weight_dict else None
        x, y = calculate_cumulative_calibration_curve(y_true_dict[part], y_pred_dict[part], weights)
        y_interp = np.interp(x_interp, x, y)
        y_interp_all.append(y_interp)

        if show_all_curves:
            ax.plot(x, y, alpha=0.3, label=f"Partition {part}")

    y_interp_all = np.array(y_interp_all)
    mean_y = np.mean(y_interp_all, axis=0)
    std_y = np.std(y_interp_all, axis=0)

    if over_and_under_colors:
        from itertools import tee
        def segment_intersects_y_eq_x(p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            f0 = y0 - x0
            f1 = y1 - x1
            if f0 * f1 >= 0:
                return None
            denom = (x1 - x0) - (y1 - y0)
            if denom == 0:
                return None
            t = (y0 - x0) / denom
            if 0 < t < 1:
                xi = x0 + t * (x1 - x0)
                yi = y0 + t * (y1 - y0)
                return (xi, yi)
            return None

        overpricing_parts = []
        underpricing_parts = []
        stack = [(x_interp[0], mean_y[0])]
        for i in range(len(x_interp) - 1):
            p0 = x_interp[i], mean_y[i]
            p1 = x_interp[i + 1], mean_y[i + 1]
            intersection = segment_intersects_y_eq_x(p0, p1)
            lays_on_eq = (p0[0] == p0[1] and p1[0] == p1[1])
            if lays_on_eq:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="black")
                stack = []
            elif intersection is not None or p1[0] == p1[1] or i == len(x_interp) - 2:
                stack.append(intersection or p1)
                if p0[1] > p0[0]:
                    overpricing_parts.append(stack)
                else:
                    underpricing_parts.append(stack)
                stack = [intersection] if intersection is not None else []
            stack.append(p1)

        for parts, color, label in zip([overpricing_parts, underpricing_parts], ["blue", "red"],
                                       ["Underpriced", "Overpriced"]):
            for part in parts:
                x = [p[0] for p in part]
                y = [p[1] for p in part]
                if fill_between_equity_and_curve:
                    ax.fill_between(x, y, x, color=color, alpha=0.05)
                ax.plot(x, y, color=color)

        legend_handles = [
            Line2D([0], [0], color="blue", label="Underpriced"),
            Line2D([0], [0], color="red", label="Overpriced")
        ]
    else:
        ax.plot(x_interp, mean_y, color="green", label="Mean Calibration Curve")
        if fill_between_equity_and_curve:
            ax.fill_between(x_interp, mean_y, y2=x_interp, color="green", alpha=0.05)
        legend_handles = [Line2D([0], [0], color="green", label="Mean Calibration Curve")]

    if show_std_band:
        color = "green"
        alpha = 0.2
        ax.fill_between(x_interp, mean_y - std_y, mean_y + std_y, color=color, alpha=alpha)
        legend_handles.append(Line2D([0], [0], color=color, alpha=alpha, linewidth=8, label="±1 Std. Dev"))

    ax.plot([0, 1], [0, 1], linestyle="dotted", color="black", label="Line of Equality")
    legend_handles.append(Line2D([0], [0], color="black", linestyle="dotted", label="Line of Equality"))

    ax.set_xlabel("Cumulative Proportion of predicted severity")
    ax.set_ylabel("Cumulative Proportion of claims sizes")
    ax.set_title(title)
    ax.legend(handles=legend_handles)
    ax.grid(True)
    plt.close(fig)
    return fig
