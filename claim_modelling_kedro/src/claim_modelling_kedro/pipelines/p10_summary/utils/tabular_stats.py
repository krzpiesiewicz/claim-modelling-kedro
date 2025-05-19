import pandas as pd
import numpy as np
from typing import List, Optional


def prediction_group_summary_strict_bins(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bins: int,
    groups: List[int] = None,
    sample_weight: Optional[pd.Series] = None,
    round_precision: int = None,
    asint: bool = False
) -> pd.DataFrame:
    bin_label_name = "group"

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    if sample_weight is not None:
        df["weight"] = sample_weight
    else:
        df["weight"] = 1.0

    # Stable argsort to break ties
    sorted_indices = np.argsort(df["y_pred"].values, kind="stable")
    group_numbers = np.zeros(len(df), dtype=int)
    for i, idx in enumerate(sorted_indices):
        group_numbers[idx] = (i * n_bins) // len(df) + 1  # 1-based group index

    df[bin_label_name] = group_numbers

    # Filter by specified groups
    if groups is not None:
        df = df[df[bin_label_name].isin(groups)]

    # Group and compute statistics
    def weighted_mean(x, w):
        return np.average(x, weights=w)

    def weighted_std(x, w):
        average = weighted_mean(x, w)
        return np.sqrt(np.average((x - average)**2, weights=w))

    def weighted_quantile(values, quantiles, sample_weight):
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_cdf /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_cdf, values)

    def round_arr(arr):
        arr = np.array(arr).astype(float)
        if round_precision is not None:
            arr = np.round(arr, round_precision)
        if asint:
            arr = arr.astype(int)
        return arr

    summary_rows = []
    for group, group_df in df.groupby(bin_label_name):
        w = group_df["weight"].values
        yt = group_df["y_true"].values
        yp = group_df["y_pred"].values

        stats = {
            "group": group,
            "n_obs": len(group_df),
            "mean_pred": round_arr(weighted_mean(yp, w)),
            "mean_target": round_arr(weighted_mean(yt, w)),
            "std_pred": round_arr(weighted_std(yp, w)),
            "std_target": round_arr(weighted_std(yt, w)),
        }

        q_vals = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        quantiles = weighted_quantile(yt, quantiles=q_vals, sample_weight=w)
        for q, val in zip(q_vals, quantiles):
            stats[f"q_{int(q*100):02d}"] = round_arr(val)

        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("group", inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df
