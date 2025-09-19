import os.path
import tempfile
import mlflow
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index, \
    load_pd_dataframe_csv_from_mlflow

logger = logging.getLogger(__name__)

FILE_NAME_PRED_GROUPS_STATS = "{dataset}_prediction_{n_bins}_group.csv"
FILE_NAME_AVERAGE_PRED_GROUPS_STATS = "{dataset}_prediction_{n_bins}_group_means.csv"

ARTIFACT_PATH_PRED_GROUPS_STATS = "prediction_group_statistics"


def _get_file_name(file_template: str, joined_dataset: str, n_bins: int) -> str:
    """
    Returns the file name based on the template and dataset name.
    Args:
        file_template (str): Template for the file name.
        joined_dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the table.
    """
    return file_template.format(dataset=joined_dataset, n_bins=n_bins)


def get_prediction_group_statistics_file_name(joined_dataset: str, n_bins: int) -> str:
    """
    Returns the file name for the prediction group statistics based on the dataset and number of bins.

    Args:
        joined_dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the table.

    Returns:
        str: Formatted file name for the prediction group statistics.
    """
    return _get_file_name(FILE_NAME_PRED_GROUPS_STATS, joined_dataset, n_bins)


def get_average_prediction_group_statistics_file_name(joined_dataset: str, n_bins: int) -> str:
    """
    Returns the file name for the average prediction group statistics based on the dataset and number of bins.

    Args:
        joined_dataset (str): Name of the dataset (e.g., "train" or "test").
        n_bins (int): Number of bins used in the table.

    Returns:
        str: Formatted file name for the average prediction group statistics.
    """
    return _get_file_name(FILE_NAME_AVERAGE_PRED_GROUPS_STATS, joined_dataset, n_bins)


def prediction_group_statistics_strict_bins(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sample_weight: Union[pd.Series, np.ndarray],
    n_bins: int,
    groups: List[int] = None,
    round_precision: int = None,
    as_int: bool = False
) -> pd.DataFrame:
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

    table_rows = []
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

        table_rows.append(stats)

    stats_df = pd.DataFrame(table_rows)
    stats_df.sort_values("group", inplace=True)
    stats_df.reset_index(drop=True, inplace=True)

    stats_df["bias_deviation"] = stats_df["mean_target"] - stats_df["mean_pred"]
    stats_df["mean_overpricing"] = np.max([stats_df["mean_target"], stats_df["mean_pred"]], axis=0) - stats_df["mean_target"]
    stats_df["mean_underpricing"] = stats_df["mean_target"] - np.min([stats_df["mean_target"], stats_df["mean_pred"]], axis=0)
    stats_df["abs_bias_deviation"] = np.abs(stats_df["bias_deviation"])

    return stats_df


def create_prediction_group_statistics_strict_bins(
    config: Config,
    predictions_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    joined_dataset: str,
    n_bins: int,
    groups: List[int] = None,
    round_precision: int = None,
    as_int: bool = False
) -> pd.DataFrame:
    """
    Generates statistics of predictions and targets grouped into bins.

    Args:
        config (Config): Configuration object containing settings and parameters.
        predictions_df (Dict[str, pd.DataFrame]): Dictionary of predictions for each partition.
        target_df (Dict[str, pd.DataFrame]): Dictionary of targets for each partition.
        prediction_col (str): Name of the column containing the predictions.
        target_col (str): Name of the column containing the target values.
        joined_dataset (str): Name of the dataset (e.g., "train" or "pure_test").
        n_bins (int): Number of bins to divide the data into.
        groups (List[int], optional): Specific groups to include in the table. Defaults to None.
        round_precision (int, optional): Precision for rounding numerical values. Defaults to None.
        as_int (bool, optional): Whether to convert rounded values to integers. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the statistics for each bin.
    """
    logger.info(f"Generating prediction group statistics for dataset: {joined_dataset} with {n_bins} bins...")

    y_true = target_df[target_col]
    y_pred = predictions_df[prediction_col]
    sample_weight = get_sample_weight(config, target_df)

    stats_df = prediction_group_statistics_strict_bins(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        n_bins=n_bins,
        groups=groups,
        round_precision=round_precision,
        as_int=as_int
    )

    # Save and log the statistics DataFrame as a CSV file to MLflow
    logger.info(f"Saving and logging the prediction group statistics for dataset: {joined_dataset} as a CSV file to MLflow...")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_prediction_group_statistics_file_name(joined_dataset, n_bins)
        artifact_path = ARTIFACT_PATH_PRED_GROUPS_STATS
        csv_path = os.path.join(temp_dir, filename)
        stats_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        logger.info(f"Prediction group statistics for dataset: {joined_dataset} logged to MLflow as {os.path.join(artifact_path, filename)}.")

    return stats_df


def create_average_prediction_group_statistics(
    config: Config,
    stats_dfs: Dict[str, pd.DataFrame],
    joined_dataset: str,
    n_bins: int,
) -> None:
    """
    Averages the prediction group statistics across partitions and logs the result to MLflow.

    Args:
        config (Config): Configuration object containing settings and parameters.
        stats_dfs (Dict[str, pd.DataFrame]): DataFrame containing the prediction group statistics.
        joined_dataset (str): Name of the dataset (e.g., "train" or "pure_test").
        n_bins (int): Number of bins used in the table.
    """
    logger.info(f"Averaging prediction group statistics for dataset: {joined_dataset} with {n_bins} bins over partitions...")
    if isinstance(stats_dfs, Dict):
        stats_dfs = stats_dfs.values()
    avg_stats_df = pd.concat(stats_dfs).groupby(level=0).mean()
    avg_stats_df["std_of_mean_pred"] = pd.concat(stats_dfs).mean_pred.groupby(level=0).std()
    avg_stats_df["std_of_mean_target"] = pd.concat(stats_dfs).mean_target.groupby(level=0).std()
    avg_stats_df["std_of_bias_deviation"] = pd.concat(stats_dfs).bias_deviation.groupby(level=0).std()
    avg_stats_df["std_of_mean_overpricing"] = pd.concat(stats_dfs).mean_overpricing.groupby(level=0).std()
    avg_stats_df["std_of_mean_underpricing"] = pd.concat(stats_dfs).mean_underpricing.groupby(level=0).std()
    avg_stats_df["std_of_abs_bias_deviation"] = pd.concat(stats_dfs).abs_bias_deviation.groupby(level=0).std()

    # Calculate relative statistics, handling division by zero by assigning np.nan
    den = avg_stats_df["mean_target"].astype(float)
    den = den.mask(np.isclose(den, 0.0), np.nan)

    for out_col, num_col in [
        ("rel_bias_deviation", "bias_deviation"),
        ("rel_mean_overpricing", "mean_overpricing"),
        ("rel_mean_underpricing", "mean_underpricing"),
        ("rel_abs_bias_deviation", "abs_bias_deviation"),
    ]:
        avg_stats_df[out_col] = avg_stats_df[num_col].astype(float).div(den)

    # Save and log the averaged statistics DataFrame as a CSV file to MLflow
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = get_average_prediction_group_statistics_file_name(joined_dataset, n_bins)
        artifact_path = ARTIFACT_PATH_PRED_GROUPS_STATS
        csv_path = os.path.join(temp_dir, filename)
        avg_stats_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        logger.info(f"Averaged prediction group statistics for dataset: {joined_dataset} over partitions logged to MLflow as {os.path.join(artifact_path, filename)}.")


def load_prediction_and_targets_group_stats_from_mlflow(
    joined_dataset: str,
    n_bins: int,
    part: str = None,
    mlflow_run_id: str = None,
    time_limit: float = None,
    raise_on_failure: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filename = get_prediction_group_statistics_file_name(joined_dataset, n_bins)
    artifact_path = ARTIFACT_PATH_PRED_GROUPS_STATS
    if part is not None:
        mlflow_run_id = get_mlflow_run_id_for_partition(partition=part, parent_mlflow_run_id=mlflow_run_id)
    stats_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id,
                                                 time_limit=time_limit, raise_on_failure=raise_on_failure)
    return stats_df


def load_averaged_prediction_group_stats_from_mlflow(
    joined_dataset: str,
    n_bins: int,
    mlflow_run_id: str = None,
    time_limit: float = None,
    raise_on_failure: bool = True
) -> pd.DataFrame:
    filename = get_average_prediction_group_statistics_file_name(joined_dataset, n_bins)
    artifact_path = ARTIFACT_PATH_PRED_GROUPS_STATS
    stats_df = load_pd_dataframe_csv_from_mlflow(artifact_path, filename, mlflow_run_id,
                                                 time_limit=time_limit, raise_on_failure=raise_on_failure)
    return stats_df
