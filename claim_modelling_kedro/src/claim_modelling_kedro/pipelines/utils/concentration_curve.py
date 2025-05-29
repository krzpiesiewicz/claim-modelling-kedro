from typing import Dict, Tuple, Optional, List
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index


def calculate_concentration_curve_parts(
        y_true: pd.Series,
        y_pred: pd.Series,
        sample_weight: pd.Series = None,
        denuit: bool = False,
        det_atol: float = 1e-9
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the concentration curve (Lorentz curve) points with merging of duplicate y_pred values
    using weighted averages for y_true and total weight as weight. Additionally, it handles colinear points
    to simplify the curve representation.

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
        sample_weight (pd.Series, optional): Observation weights. If None, all weights are set to 1.
        denuit (bool): If True, modifies the curve calculation to use CC[Y, m^(X); α] = E[Y⋅1{m^(X)≤Fm^(X)−1 (α)}] / E[Y].
                       This is useful for specific statistical applications.
        det_atol (float): Tolerance for determining colinear points. Points are considered colinear if the determinant
                          of their coordinates is below this threshold.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of merged curve segments, where each segment is represented as
                                             (x_points, y_points) arrays.

    Notes:
        - The function assumes that `ordered_by_pred_and_hashed_index` sorts y_pred in ascending order and breaks ties
          randomly based on the index.
        - Colinear points are merged to simplify the curve representation.
    """
    # Sort data by y_pred and handle ties using hashed index
    y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)

    # Convert inputs to NumPy arrays for faster processing
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.ones_like(y_true) if sample_weight is None else np.asarray(sample_weight)

    # Buffers for merged groups
    merged_y_true = []
    merged_weight = []
    merged_number = []

    # Initialize the first group
    current_pred = y_pred[0]
    weight_sum = 0
    weighted_y_true_sum = 0

    def save_current_group():
        """
        Save the current group by appending the weighted sum of y_true and the total weight.
        """
        merged_y_true.append(weighted_y_true_sum)
        merged_weight.append(weight_sum)

    # Iterate through predictions to group and merge values
    for i in range(len(y_pred)):
        weighted_true = y_true[i] * sample_weight[i]
        if y_pred[i] != current_pred:
            # Save the current group
            save_current_group()
            # Start a new group
            current_pred = y_pred[i]
            weight_sum = 0
            weighted_y_true_sum = 0
        # Continue the current group
        weight_sum += sample_weight[i]
        weighted_y_true_sum += weighted_true

        # Save the last group
    save_current_group()

    # Normalize merged values
    merged_y_true = np.insert(np.array(merged_y_true) / sum(merged_y_true), 0, 0)
    merged_weight = np.insert(np.array(merged_weight) / sum(merged_weight), 0, 0)

    # Calculate cumulative sums
    cum_true = np.cumsum(merged_y_true)
    cum_weight = np.cumsum(merged_weight)

    # Create curve intervals
    intervals = []
    for i in range(1, len(merged_y_true)):
        alpha0 = cum_weight[i - 1]
        alpha1 = cum_weight[i]
        gamma0 = cum_true[i] if denuit else cum_true[i - 1]
        gamma1 = cum_true[i]
        intervals.append(((alpha0, gamma0), (alpha1, gamma1)))

    def are_colinear(x0, x1, x2, y0, y1, y2) -> bool:
        """
        Check if three points are colinear using the determinant of their coordinates.

        Args:
            x0, x1, x2 (float): X-coordinates of the points.
            y0, y1, y2 (float): Y-coordinates of the points.

        Returns:
            bool: True if the points are colinear, False otherwise.
        """
        det = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
        return abs(det) < det_atol

    # Add a dummy interval for processing
    intervals.append(((1, 0), (2, -2)))

    # Merge colinear intervals
    merged_colinear = []
    (alpha0, gamma0), (alpha1, gamma1) = intervals[0]
    alpha2, gamma2 = alpha1, gamma1
    for i in range(len(intervals)):
        (alpha3, gamma3), (alpha4, gamma4) = intervals[i]

        if not (are_colinear(alpha0, alpha1, alpha3, gamma0, gamma1, gamma3)
                and are_colinear(alpha0, alpha1, alpha4, gamma0, gamma1, gamma4)):
            merged_colinear.append(([alpha0, alpha2], [gamma0, gamma2]))
            (alpha0, gamma0), (alpha1, gamma1) = (alpha3, gamma3), (alpha4, gamma4)
        alpha2, gamma2 = alpha4, gamma4

    # Combine consecutive segments into curves
    merged_curves = []
    current_xs = merged_colinear[0][0].copy()
    current_ys = merged_colinear[0][1].copy()
    for i in range(1, len(merged_colinear)):
        xs, ys = merged_colinear[i]
        if current_xs[-1] == xs[0] and current_ys[-1] == ys[0]:
            current_xs.append(xs[1])
            current_ys.append(ys[1])
        else:
            merged_curves.append((np.array(current_xs), np.array(current_ys)))
            current_xs = xs.copy()
            current_ys = ys.copy()
    merged_curves.append((np.array(current_xs), np.array(current_ys)))

    return merged_curves


def join_calculate_concentration_curve_parts(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Join multiple concentration curve parts into a single curve.

    Args:
        curves (List[Tuple[np.ndarray, np.ndarray]]): List of tuples containing x and y points of the curves.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Combined x and y points of the concentration curve.
    """
    x_points = np.concatenate([curve[0] for curve in curves])
    y_points = np.concatenate([curve[1] for curve in curves])
    
    # Remove duplicates while preserving order
    unique_indices = np.unique(x_points, return_index=True)[1]
    sorted_indices = np.argsort(unique_indices)
    
    return x_points[sorted_indices], y_points[sorted_indices]


def calculate_concentration_curve(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series = None,
    det_atol: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the concentration curve (Lorentz curve) by joining its parts.

    Args:
        y_true (pd.Series): Series of true values.
        y_pred (pd.Series): Series of predicted values.
        sample_weight (pd.Series, optional): Series of sample weights. If None, all weights are set to 1.
        det_atol (float): Tolerance for determining colinear points. Points are considered colinear if the determinant
                          of their coordinates is below this threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays representing the x and y coordinates of the joined concentration curve.

    Notes:
        - The function first calculates the individual parts of the concentration curve using `calculate_concentration_curve_parts`.
        - Then, it joins these parts into a single curve using `join_calculate_concentration_curve_parts`.
    """
    curve_parts = calculate_concentration_curve_parts(y_true, y_pred, sample_weight, det_atol=det_atol)
    return join_calculate_concentration_curve_parts(curve_parts)


def interpolate_to_points(
    xs: np.ndarray[float],
    ys: np.ndarray[float],
    x_interp: np.ndarray[float],
    trunc_to_xs: bool = True
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Interpolates y-values for given x-values using linear interpolation.

    Args:
        xs (np.ndarray[float]): Array of x-values (must be sorted in ascending order).
        ys (np.ndarray[float]): Array of corresponding y-values.
        x_interp (np.ndarray[float]): Array of x-values for interpolation.
        trunc_to_xs (bool): If True, restricts `x_interp` to the range of `xs`. Defaults to True.

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: Interpolated x-values and corresponding y-values.

    Notes:
        - If `trunc_to_xs` is True, `x_interp` is truncated to the range `[xs[0], xs[-1]]`.
        - Uses `np.interp` for efficient linear interpolation.
    """
    if trunc_to_xs:
        # Restrict interpolation points to the range of xs
        x_interp = x_interp[(x_interp >= xs[0]) & (x_interp <= xs[-1])]
    # Perform linear interpolation
    y_interp = np.interp(x_interp, xs, ys)
    return x_interp, y_interp


def get_common_x_interp(curves_parts: List[List[Tuple[np.ndarray[float], np.ndarray[float]]]], n_points_per_one: int = 200) -> np.ndarray[float]:
    """
    Generates a common set of x-values for interpolation across multiple curve parts.

    Args:
        curves_parts (List[List[Tuple[np.ndarray[float], np.ndarray[float]]]]): List of curve parts, where each part is a tuple of x and y arrays.
        n_points_per_one (int): Number of evenly spaced points between 0 and 1 for interpolation. Defaults to 200.

    Returns:
        np.ndarray[float]: Sorted array of unique x-values for interpolation.

    Notes:
        - Extracts the start and end x-values from each curve part.
        - Combines these values with evenly spaced points between 0 and 1.
        - Ensures the resulting array contains unique and sorted x-values.
    """
    xs = []
    # Collect start and end x-values from all curve parts
    for parts in curves_parts:
        for xs_part, _ in parts:
            xs.append(xs_part[0])  # Start of the curve part
            xs.append(xs_part[-1])  # End of the curve part
    # Generate evenly spaced x-values
    x_interp = np.linspace(0, 1, n_points_per_one)
    # Combine and sort unique x-values
    x_interp = np.concatenate([x_interp, xs])
    return np.sort(np.unique(x_interp))


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
        n_points: int = 200,  # Number of interpolation points if `denuit` is False
        denuit: bool = False  # Whether to use the Denuit method for curve calculation
) -> plt.Figure:
    """
    Generates a concentration curve plot based on true and predicted values, with support for multiple curve types.

    Args:
        y_true (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
        sample_weight (Optional[pd.Series]): Weights for observations (optional). Defaults to None.
        title (str): Title of the plot. Defaults to "Concentration Curve".
        marker (Optional[str]): Marker style for the plot points. Defaults to None.
        show_cc (bool): Whether to display the concentration curve. Defaults to True.
        show_oracle (bool): Whether to display the oracle curve (ideal curve). Defaults to False.
        fill_between_cc_and_oracle (bool): Whether to fill the area between the concentration curve and the oracle curve. Defaults to False.
        fill_between_equity_and_cc (bool): Whether to fill the area between the line of equality and the concentration curve. Defaults to True.
        show_std_band (bool): Placeholder for API compatibility (not used in this function). Defaults to False.
        show_all_curves (bool): Placeholder for API compatibility (not used in this function). Defaults to False.
        show_lorenz (bool): Whether to display the Lorenz curve. Defaults to False.
        fill_between_cc_and_lorenz (bool): Whether to fill the area between the concentration curve and the Lorenz curve. Defaults to False.
        fill_under_lorenz (bool): Whether to fill the area under the Lorenz curve. Defaults to False.
        n_points (int): Number of interpolation points for the curves. Defaults to 200.
        denuit (bool): Whether to use the Denuit method for curve calculation. Defaults to True.

    Returns:
        plt.Figure: A matplotlib figure object containing the concentration curve plot.

    Notes:
        - The function supports plotting multiple types of curves: concentration curve, oracle curve, and Lorenz curve.
        - Interpolation is used to ensure all curves share a common set of x-points for comparison.
        - Areas between curves can be filled for visualization purposes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # List to store legend handles for the plot
    legend_handles = []

    # Calculate curve parts for each type of curve
    cc_parts = calculate_concentration_curve_parts(y_true, y_pred, sample_weight, denuit=denuit) if show_cc else []
    oc_parts = calculate_concentration_curve_parts(y_true, y_true, sample_weight, denuit=denuit) if show_oracle else []
    lc_parts = calculate_concentration_curve_parts(y_pred, y_pred, sample_weight, denuit=denuit) if show_lorenz else []

    # Generate common x-points for interpolation across all curves
    common_x_interp = get_common_x_interp([cc_parts, oc_parts, lc_parts], n_points_per_one=n_points)

    # Plot the concentration curve (CC)
    if show_cc:
        for i in range(len(cc_parts)):
            x_interp, y_interp = interpolate_to_points(*cc_parts[i], x_interp=common_x_interp)
            handle, = ax.plot(x_interp, y_interp, color="blue", marker=marker)
            if i == 0:  # Add label for the first segment
                handle.set_label("Concentration Curve")
                legend_handles.append(handle)
        # Join and interpolate the full concentration curve
        cc = join_calculate_concentration_curve_parts(cc_parts)
        cc_xs, cc_ys = interpolate_to_points(*cc, x_interp=common_x_interp, trunc_to_xs=False)
        if fill_between_equity_and_cc:
            ax.fill_between(cc_xs, cc_ys, y2=cc_xs, color="blue", alpha=0.05)
    else:
        cc_xs, cc_ys = None, None

    # Plot the oracle curve (OC)
    if show_oracle:
        for i in range(len(oc_parts)):
            x_interp, y_interp = interpolate_to_points(*oc_parts[i], x_interp=common_x_interp)
            handle, = ax.plot(x_interp, y_interp, linestyle="--", color="black")
            if i == 0:  # Add label for the first segment
                handle.set_label("Oracle Curve")
                legend_handles.append(handle)
        if fill_between_cc_and_oracle and cc_ys is not None:
            oc = join_calculate_concentration_curve_parts(oc_parts)
            oc_xs, oc_ys = interpolate_to_points(*oc, x_interp=common_x_interp, trunc_to_xs=False)
            assert all(oc_xs == cc_xs)  # Ensure x-points match
            ax.fill_between(common_x_interp, oc_ys, cc_ys, color="gray", alpha=0.1)

    # Plot the Lorenz curve (LC)
    if show_lorenz:
        for i in range(len(lc_parts)):
            x_interp, y_interp = interpolate_to_points(*lc_parts[i], x_interp=common_x_interp)
            lc_parts[i] = (x_interp, y_interp)
            handle, = ax.plot(x_interp, y_interp, linestyle="-.", color="green")
            if i == 0:  # Add label for the first segment
                handle.set_label("Lorenz Curve")
                legend_handles.append(handle)
        lc = join_calculate_concentration_curve_parts(lc_parts)
        lc_xs, lc_ys = interpolate_to_points(*lc, x_interp=common_x_interp, trunc_to_xs=False)
        if fill_under_lorenz:
            ax.fill_between(lc_xs, 0, lc_ys, color="green", alpha=0.05)
        if fill_between_cc_and_lorenz:
            ax.fill_between(lc_xs, lc_ys, cc_ys, color="red", alpha=0.05)

    # Add the line of equality
    ax.plot([0, 1], [0, 1], label="Line of Equality", linestyle="dotted", color="black")

    # Set axis labels and title
    ax.set_xlabel("Cumulative Proportion of Observations (Ordered by Predicted Values)")
    update_plot_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorenz)
    ax.set_title(title)

    # Add legend and grid
    ax.legend(handles=legend_handles + [ax.lines[-1]])  # Include the line of equality in the legend
    ax.grid(True)

    # Close the figure to prevent display in interactive environments
    plt.close(fig)
    return fig


# def plot_concentration_curve(
#     y_true: pd.Series,
#     y_pred: pd.Series,
#     sample_weight: Optional[pd.Series] = None,
#     title: str = "Concentration Curve",
#     marker: Optional[str] = None,
#     show_cc: bool = True,
#     show_oracle: bool = False,
#     fill_between_cc_and_oracle: bool = False,
#     fill_between_equity_and_cc: bool = True,
#     show_std_band: bool = False,  # Placeholder for API compatibility
#     show_all_curves: bool = False,  # Placeholder for API compatibility
#     show_lorenz: bool = False,
#     fill_between_cc_and_lorenz: bool = False,
#     fill_under_lorenz: bool = False,
#     n_points: int = 100
# ) -> plt.Figure:
#     """
#     Generates a concentration curve plot based on true and predicted values.
#
#     Args:
#         y_true (pd.Series): True target values.
#         y_pred (pd.Series): Predicted target values.
#         sample_weight (Optional[pd.Series]): Weights for observations (optional).
#         title (str): Title of the plot.
#         marker (Optional[str]): Marker style for the plot points.
#         show_cc (bool): Whether to display the concentration curve.
#         show_oracle (bool): Whether to display the oracle curve (ideal curve).
#         fill_between_cc_and_oracle (bool): Whether to fill the area between the concentration curve and the oracle curve.
#         fill_between_equity_and_cc (bool): Whether to fill the area between the line of equality and the concentration curve.
#         show_std_band (bool): Placeholder for API compatibility (not used in this function).
#         show_all_curves (bool): Placeholder for API compatibility (not used in this function).
#         show_lorenz (bool): Whether to display the Lorenz curve.
#         fill_between_cc_and_lorenz (bool): Whether to fill the area between the concentration curve and the Lorenz curve.
#         fill_under_lorenz (bool): Whether to fill the area under the Lorenz curve.
#         n_points (int): Number of interpolation points for the curves.
#
#     Returns:
#         plt.Figure: A matplotlib figure object containing the concentration curve plot.
#     """
#     x_interp = np.linspace(0, 1, n_points)
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     if show_cc:
#         x_cc, y_cc = calculate_concentration_curve(y_true, y_pred, sample_weight)
#         y_cc_interp = np.interp(x_interp, x_cc, y_cc)
#         ax.plot(x_interp, y_cc_interp, label="Concentration Curve", color="blue", marker=marker)
#         if fill_between_equity_and_cc:
#             ax.fill_between(x_interp, y_cc_interp, y2=x_interp, color="blue", alpha=0.05)
#     else:
#         y_cc_interp = None
#
#     if show_oracle:
#         x_oracle, y_oracle = calculate_concentration_curve(y_true, y_true, sample_weight)
#         y_oracle_interp = np.interp(x_interp, x_oracle, y_oracle)
#         ax.plot(x_interp, y_oracle_interp, label="Oracle Curve", linestyle="--", color="black")
#         if fill_between_cc_and_oracle and y_cc_interp is not None:
#             ax.fill_between(x_interp, y_oracle_interp, y_cc_interp, color="gray", alpha=0.05)
#
#     if show_lorenz:
#         x_lorenz, y_lorenz = calculate_concentration_curve(y_pred, y_pred, sample_weight)
#         y_lorenz_interp = np.interp(x_interp, x_lorenz, y_lorenz)
#         ax.plot(x_interp, y_lorenz_interp, label="Lorenz Curve", linestyle="-.", color="green")
#         if fill_between_cc_and_lorenz and y_cc_interp is not None:
#             ax.fill_between(x_interp, y_cc_interp, y_lorenz_interp, color="green", alpha=0.05)
#         if fill_under_lorenz:
#             ax.fill_between(x_interp, 0, y_lorenz_interp, color="green", alpha=0.03)
#
#     ax.plot([0, 1], [0, 1], label="Line of Equality", linestyle="dotted", color="black")
#     ax.set_xlabel("Cumulative Proportion of Observations (Ordered by Predicted Values)")
#     update_plot_concentration_curve_axis_label(ax, show_cc, show_oracle, show_lorenz)
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True)
#     plt.close(fig)
#     return fig


def plot_mean_concentration_curve(
    y_true_dict: Dict[str, pd.Series],
    y_pred_dict: Dict[str, pd.Series],
    sample_weight_dict: Dict[str, pd.Series] = None,
    title: str = "Mean Concentration Curve across Partitions",
    marker: str = None,
    show_cc: bool = True,
    show_oracle: bool = False,
    fill_between_cc_and_oracle: bool = False,
    fill_between_equity_and_cc: bool = False,
    show_std_band: bool = True,
    show_all_curves: bool = False,
    show_lorenz: bool = False,
    fill_between_cc_and_lorenz: bool = False,
    fill_under_lorenz: bool = False,
    n_points: int = 200
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
