from typing import Dict, Tuple, Optional
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


def calculate_concentration_curve(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the points for the Lorentz curve based on true and predicted values, considering observation weights.

    Args:
        y_true (np.ndarray or pd.Series): True values.
        y_pred (np.ndarray or pd.Series): Predicted values.
        sample_weight (np.ndarray or pd.Series, optional): Weights for observations. Defaults to equal weights.

    Returns:
        tuple: (x_points, y_points) for the weighted Lorentz curve.
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Initialize equal weights if not provided
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    else:
        sample_weight = np.asarray(sample_weight)

    # Combine true values, predicted values, and weights into a single array
    data = np.c_[y_true, y_pred, sample_weight]

    # Sort by predicted values in ascending order (to construct cumulative distribution)
    data_sorted = data[np.argsort(data[:, 1])]

    # Weighted cumulative sum of true values
    weighted_true = data_sorted[:, 0] * data_sorted[:, 2]  # y_true * sample_weight
    cumulative_weighted_true = np.cumsum(weighted_true)

    # Total weighted true value
    total_weighted_true = np.sum(weighted_true)

    # Weighted cumulative count of observations
    cumulative_weighted_count = np.cumsum(data_sorted[:, 2])
    total_weighted_count = np.sum(data_sorted[:, 2])

    # Calculate Lorentz curve points
    x_points = np.insert(cumulative_weighted_count / total_weighted_count, 0, 0)  # Proportion of total weighted observations
    y_points = np.insert(cumulative_weighted_true / total_weighted_true, 0, 0)  # Proportion of total weighted true values

    return x_points, y_points


def determine_y_axis_label(show_cc: bool, show_oracle: bool, show_lorentz: bool) -> str:
    """
    Determine the Y-axis label based on which curves are shown.
    """
    if (not show_cc and not show_oracle) and show_lorentz:
        return "Cumulative Proportion of Predicted Values"
    elif (show_cc or show_oracle) and not show_lorentz:
        return "Cumulative Proportion of True Values"
    elif (show_cc or show_oracle) and show_lorentz:
        return "Cumulative Proportion of True / Predicted Values"
    else:
        return "Cumulative Proportion"


def update_plot_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorentz):
    """
    Updates the Y-axis label for the concentration curve plot.

    Args:
        ax: The matplotlib axis object to update.
        show_cc (bool): Indicates if the concentration curve is displayed.
        show_oracle (bool): Indicates if the oracle curve is displayed.
        show_lorentz (bool): Indicates if the Lorenz curve is displayed.
    """
    label = determine_y_axis_label(show_cc, show_oracle, show_lorentz)
    ax.set_ylabel(label)


def update_plot_mean_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorentz):
    """
    Updates the Y-axis label for the mean concentration curve plot.

    Args:
        ax: The matplotlib axis object to update.
        show_cc (bool): Indicates if the concentration curve is displayed.
        show_oracle (bool): Indicates if the oracle curve is displayed.
        show_lorentz (bool): Indicates if the Lorenz curve is displayed.
    """
    label = determine_y_axis_label(show_cc, show_oracle, show_lorentz)
    ax.set_ylabel(label)


def plot_concentration_curve(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    title: str = "Concentration Curve",
    marker: Optional[str] = None,
    show_cc: bool = True,
    show_oracle: bool = False,
    fill_between_cc_and_oracle: bool = False,
    fill_between_equity_and_cc: bool = True,
    show_std_band: bool = False,  # Placeholder for API compatibility
    show_all_curves: bool = False,  # Placeholder for API compatibility
    show_lorenz: bool = False,
    fill_between_cc_and_lorenz: bool = False,
    fill_under_lorenz: bool = False,
    n_points: int = 100
) -> plt.Figure:
    """
    Generates a concentration curve plot based on true and predicted values.

    Args:
        y_true (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
        sample_weight (Optional[pd.Series]): Weights for observations (optional).
        title (str): Title of the plot.
        marker (Optional[str]): Marker style for the plot points.
        show_cc (bool): Whether to display the concentration curve.
        show_oracle (bool): Whether to display the oracle curve (ideal curve).
        fill_between_cc_and_oracle (bool): Whether to fill the area between the concentration curve and the oracle curve.
        fill_between_equity_and_cc (bool): Whether to fill the area between the line of equality and the concentration curve.
        show_std_band (bool): Placeholder for API compatibility (not used in this function).
        show_all_curves (bool): Placeholder for API compatibility (not used in this function).
        show_lorenz (bool): Whether to display the Lorenz curve.
        fill_between_cc_and_lorenz (bool): Whether to fill the area between the concentration curve and the Lorenz curve.
        fill_under_lorenz (bool): Whether to fill the area under the Lorenz curve.
        n_points (int): Number of interpolation points for the curves.

    Returns:
        plt.Figure: A matplotlib figure object containing the concentration curve plot.
    """
    x_interp = np.linspace(0, 1, n_points)
    fig, ax = plt.subplots(figsize=(8, 6))

    if show_cc:
        x_cc, y_cc = calculate_concentration_curve(y_true, y_pred, sample_weight)
        y_cc_interp = np.interp(x_interp, x_cc, y_cc)
        ax.plot(x_interp, y_cc_interp, label="Concentration Curve", color="blue", marker=marker)
        if fill_between_equity_and_cc:
            ax.fill_between(x_interp, y_cc_interp, y2=x_interp, color="blue", alpha=0.05)
    else:
        y_cc_interp = None

    if show_oracle:
        x_oracle, y_oracle = calculate_concentration_curve(y_true, y_true, sample_weight)
        y_oracle_interp = np.interp(x_interp, x_oracle, y_oracle)
        ax.plot(x_interp, y_oracle_interp, label="Oracle Curve", linestyle="--", color="black")
        if fill_between_cc_and_oracle and y_cc_interp is not None:
            ax.fill_between(x_interp, y_oracle_interp, y_cc_interp, color="gray", alpha=0.05)

    if show_lorenz:
        x_lorenz, y_lorenz = calculate_concentration_curve(y_pred, y_pred, sample_weight)
        y_lorenz_interp = np.interp(x_interp, x_lorenz, y_lorenz)
        ax.plot(x_interp, y_lorenz_interp, label="Lorenz Curve", linestyle="-.", color="green")
        if fill_between_cc_and_lorenz and y_cc_interp is not None:
            ax.fill_between(x_interp, y_cc_interp, y_lorenz_interp, color="green", alpha=0.05)
        if fill_under_lorenz:
            ax.fill_between(x_interp, 0, y_lorenz_interp, color="green", alpha=0.03)

    ax.plot([0, 1], [0, 1], label="Line of Equality", linestyle="dotted", color="black")
    ax.set_xlabel("Cumulative Proportion of Observations (Ordered by Predicted Values)")
    update_plot_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorenz)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.close(fig)
    return fig


def plot_mean_concentration_curve(
    y_true_dict: Dict[str, pd.Series],
    y_pred_dict: Dict[str, pd.Series],
    sample_weight_dict: Dict[str, pd.Series] = None,
    title: str = "Mean Concentration Curve across Partitions",
    marker: str = None,
    show_cc: bool = True,
    show_oracle: bool = False,
    fill_between_cc_and_oracle: bool = False,
    fill_between_equity_and_cc: bool = True,
    show_std_band: bool = True,
    show_all_curves: bool = False,
    show_lorenz: bool = False,
    fill_between_cc_and_lorenz: bool = False,
    fill_under_lorenz: bool = False,
    n_points: int = 100
) -> plt.Figure:
    """
    Generates a plot of the mean concentration curve across multiple partitions.

    Args:
        y_true_dict (Dict[str, pd.Series]): Dictionary of true target values for each partition.
        y_pred_dict (Dict[str, pd.Series]): Dictionary of predicted target values for each partition.
        sample_weight_dict (Dict[str, pd.Series], optional): Dictionary of sample weights for each partition. Defaults to None.
        title (str): Title of the plot.
        marker (str, optional): Marker style for the plot points. Defaults to None.
        show_cc (bool): Whether to display the concentration curve. Defaults to True.
        show_oracle (bool): Whether to display the oracle curve (ideal curve). Defaults to False.
        fill_between_cc_and_oracle (bool): Whether to fill the area between the concentration curve and the oracle curve. Defaults to False.
        fill_between_equity_and_cc (bool): Whether to fill the area between the line of equality and the concentration curve. Defaults to True.
        show_std_band (bool): Whether to display the standard deviation band around the mean curve. Defaults to True.
        show_all_curves (bool): Whether to display all individual curves for each partition. Defaults to False.
        show_lorenz (bool): Whether to display the Lorenz curve. Defaults to False.
        fill_between_cc_and_lorenz (bool): Whether to fill the area between the concentration curve and the Lorenz curve. Defaults to False.
        fill_under_lorenz (bool): Whether to fill the area under the Lorenz curve. Defaults to False.
        n_points (int): Number of interpolation points for the curves. Defaults to 100.

    Returns:
        plt.Figure: A matplotlib figure object containing the mean concentration curve plot.
    """
    x_interp = np.linspace(0, 1, n_points)
    y_interp_all, oracle_interp_all, lorenz_interp_all = [], [], []

    for part in y_true_dict:
        weights = sample_weight_dict.get(part) if sample_weight_dict else None

        if show_cc:
            x, y = calculate_concentration_curve(y_true_dict[part], y_pred_dict[part], weights)
            y_interp = np.interp(x_interp, x, y)
            y_interp_all.append(y_interp)
            if show_all_curves:
                plt.plot(x, y, label=f"CC for {part}", alpha=0.3)

        if show_oracle:
            xo, yo = calculate_concentration_curve(y_true_dict[part], y_true_dict[part], weights)
            oracle_interp = np.interp(x_interp, xo, yo)
            oracle_interp_all.append(oracle_interp)

        if show_lorenz:
            xl, yl = calculate_concentration_curve(y_pred_dict[part], y_pred_dict[part], weights)
            lorenz_interp = np.interp(x_interp, xl, yl)
            lorenz_interp_all.append(lorenz_interp)

    fig, ax = plt.subplots(figsize=(8, 6))

    if show_cc and y_interp_all:
        y_interp_all = np.array(y_interp_all)
        mean_y = np.mean(y_interp_all, axis=0)
        std_y = np.std(y_interp_all, axis=0)
        ax.plot(x_interp, mean_y, label="Mean Concentration Curve", color="blue", marker=marker)
        if show_std_band:
            ax.fill_between(x_interp, mean_y - std_y, mean_y + std_y, color="blue", alpha=0.1, label="±1 Std. Dev.")
        if fill_between_equity_and_cc:
            ax.fill_between(x_interp, mean_y, y2=x_interp, color="blue", alpha=0.05)

    if show_oracle and oracle_interp_all:
        oracle_interp_all = np.array(oracle_interp_all)
        mean_oracle_y = np.mean(oracle_interp_all, axis=0)
        std_oracle_y = np.std(oracle_interp_all, axis=0)
        ax.plot(x_interp, mean_oracle_y, label="Mean Oracle Curve", color="black", linestyle="--")
        if show_std_band:
            ax.fill_between(x_interp, mean_oracle_y - std_oracle_y, mean_oracle_y + std_oracle_y,
                            color="gray", alpha=0.1, label="Oracle ±1 Std. Dev.")
        if fill_between_cc_and_oracle and show_cc:
            ax.fill_between(x_interp, mean_oracle_y, mean_y, color="gray", alpha=0.05)

    if show_lorenz and lorenz_interp_all:
        lorenz_interp_all = np.array(lorenz_interp_all)
        mean_lorenz_y = np.mean(lorenz_interp_all, axis=0)
        std_lorenz_y = np.std(lorenz_interp_all, axis=0)
        ax.plot(x_interp, mean_lorenz_y, label="Mean Lorenz Curve", linestyle="-.", color="green")
        if show_std_band:
            ax.fill_between(x_interp, mean_lorenz_y - std_lorenz_y, mean_lorenz_y + std_lorenz_y,
                            color="green", alpha=0.1, label="Lorenz ±1 Std. Dev.")
        if fill_between_cc_and_lorenz and show_cc:
            ax.fill_between(x_interp, mean_y, mean_lorenz_y, color="green", alpha=0.05)
        if fill_under_lorenz:
            ax.fill_between(x_interp, 0, mean_lorenz_y, color="green", alpha=0.03)

    ax.plot([0, 1], [0, 1], label="Line of Equality", linestyle="dotted", color="black")
    ax.set_xlabel("Cumulative Proportion of Observations (Ordered by Predicted Values)")
    update_plot_mean_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorenz)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.close(fig)
    return fig
