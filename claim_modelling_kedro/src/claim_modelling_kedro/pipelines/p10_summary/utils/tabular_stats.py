import os.path
import tempfile
import mlflow
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index

logger = logging.getLogger(__name__)

FILE_NAME_PRED_GROUPS_SUMMARY = "{dataset}_prediction_{n_bins}_group_summary.csv"
FILE_NAME_AVERAGE_PRED_GROUPS_SUMMARY = "{dataset}_prediction_{n_bins}_group_summary_means.csv"

ARTIFACT_PATH_PRED_GROUPS_SUMMARY = "summary/prediction_group_summary"


def get_file_name(file_template: str, dataset: str, n_bins: int) -> str:
    """
    Returns the file name based on the template and dataset name.
    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the summary.
    """
    return file_template.format(dataset=dataset, n_bins=n_bins)


def create_prediction_group_summary_strict_bins(
    config: Config,
    predictions_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    dataset: str,
    n_bins: int,
    prefix: str = None,
    groups: List[int] = None,
    round_precision: int = None,
    as_int: bool = False
) -> pd.DataFrame:
    """
    Generates a summary of predictions and targets grouped into bins.

    Args:
        config (Config): Configuration object containing settings and parameters.
        predictions_df (Dict[str, pd.DataFrame]): Dictionary of predictions for each partition.
        target_df (Dict[str, pd.DataFrame]): Dictionary of targets for each partition.
        prediction_col (str): Name of the column containing the predictions.
        target_col (str): Name of the column containing the target values.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins to divide the data into.
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
        groups (List[int], optional): Specific groups to include in the summary. Defaults to None.
        round_precision (int, optional): Precision for rounding numerical values. Defaults to None.
        as_int (bool, optional): Whether to convert rounded values to integers. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the summary statistics for each bin.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Generating prediction group summary for dataset: {dataset} with {n_bins} bins...")

    y_true = target_df[target_col]
    y_pred = predictions_df[prediction_col]
    sample_weight = get_sample_weight(config, target_df)
    y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "weight": sample_weight
    })

    bin_label_name = "group"
    group_numbers = np.zeros(len(df), dtype=int)
    for i, idx in enumerate(df.index):
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
        return np.sqrt(np.average((x - average) ** 2, weights=w))

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
        if as_int:
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
            stats[f"q_{int(q * 100):02d}"] = round_arr(val)

        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("group", inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    # Save and log the summary DataFrame as a CSV file to MLflow
    logger.info(f"Saving and logging the prediction group summary for dataset: {dataset} as a CSV file to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_PRED_GROUPS_SUMMARY, dataset, n_bins)
        artifact_path = ARTIFACT_PATH_PRED_GROUPS_SUMMARY
        csv_path = os.path.join(temp_dir, filename)
        summary_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        logger.info(f"Prediction group summary for dataset: {dataset} logged to MLflow as {os.path.join(artifact_path, filename)}.")

    return summary_df


def create_average_prediction_group_summary(
    config: Config,
    summary_df: Dict[str, pd.DataFrame],
    dataset: str,
    n_bins: int,
    prefix: str = None
) -> None:
    """
    Averages the prediction group summary statistics across partitions and logs the result to MLflow.

    Args:
        config (Config): Configuration object containing settings and parameters.
        stats_df (pd.DataFrame): DataFrame containing the prediction group summary statistics.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the summary.
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
    """
    dataset = f"{prefix}_{dataset}" if prefix is not None else dataset
    logger.info(f"Averaging prediction group summary for dataset: {dataset} with {n_bins} bins over partitions...")
    if isinstance(summary_df, Dict):
        summary_df = summary_df.values()
    for df in summary_df:
        df["bias_deviation"] = df["mean_target"] - df["mean_pred"]
        df["mean_overpricing"] = np.max([df["mean_target"], df["mean_pred"]], axis=0) - df["mean_target"]
        df["mean_underpricing"] = df["mean_target"] - np.min([df["mean_target"], df["mean_pred"]], axis=0)
    avg_stats_df = pd.concat(summary_df).groupby(level=0).mean()
    avg_stats_df["std_of_mean_pred"] = pd.concat(summary_df).mean_pred.groupby(level=0).std()
    avg_stats_df["std_of_mean_target"] = pd.concat(summary_df).mean_target.groupby(level=0).std()
    avg_stats_df["std_of_bias_deviation"] = pd.concat(summary_df).bias_deviation.groupby(level=0).std()
    avg_stats_df["std_of_mean_overpricing"] = pd.concat(summary_df).mean_overpricing.groupby(level=0).std()
    avg_stats_df["std_of_mean_underpricing"] = pd.concat(summary_df).mean_underpricing.groupby(level=0).std()

    # Save and log the averaged summary DataFrame as a CSV file to MLflow
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_file_name(FILE_NAME_AVERAGE_PRED_GROUPS_SUMMARY, dataset, n_bins)
        artifact_path = ARTIFACT_PATH_PRED_GROUPS_SUMMARY
        csv_path = os.path.join(temp_dir, filename)
        avg_stats_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        logger.info(f"Averaged prediction group summary for dataset: {dataset} over partitions logged to MLflow as {os.path.join(artifact_path, filename)}.")
