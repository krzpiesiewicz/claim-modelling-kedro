from functools import partial
from typing import Dict, Tuple, List

import logging
import mlflow
import numpy as np
import pandas as pd
from tabulate import tabulate

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum
from claim_modelling_kedro.pipelines.p07_data_science.tabular_stats import N_BINS_LIST, \
    create_prediction_group_statistics_strict_bins, create_average_prediction_group_statistics
from claim_modelling_kedro.pipelines.utils.dataframes import save_metrics_table_in_mlflow, \
    save_metrics_cv_stats_in_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.metrics import get_metric_from_enum


logger = logging.getLogger(__name__)


def evaluate_predictions_part(config: Config, predictions_df: pd.DataFrame,
                              target_df: pd.DataFrame, dataset: str, prefix: str, part: str, log_metrics_to_mlflow: bool,
                              log_metrics_to_console: bool = True, compute_group_stats: bool = True,
                              keys: pd.Index = None) -> Tuple[
    Dict[Tuple[MetricEnum, str], float], Dict[int, pd.DataFrame]]:
    """
    Evaluates the predictions for a specific partition of the dataset.
    Args:
        config (Config): Configuration object containing settings and parameters.
        predictions_df (pd.DataFrame): DataFrame containing the predictions for the partition.
        target_df (pd.DataFrame): DataFrame containing the target values for the partition.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str): Prefix for the dataset ('pure' or None).
        part (str): Name of the partition (e.g., "train", "valid", "test").
        log_metrics_to_mlflow (bool): Whether to log metrics to MLflow.
        log_metrics_to_console (bool): Whether to log metrics to console. Defaults to True.
        keys (pd.Index, optional): Index of keys to filter the predictions and target DataFrames.
    Returns:
        Tuple[Dict[Tuple[MetricEnum, str], float], pd.DataFrame]: A tuple containing:
            - scores: Dictionary of scores for each metric.
            - stats_df_per_n_bins: DataFrame containing the statistics for the predictions and targets grouped into bins.
    """
    joined_dataset = f"{dataset}_{prefix}"
    logger.info(f"Evaluating the predictions for partition '{part}' of {joined_dataset} dataset...")
    scores = {}
    if compute_group_stats:
        logger.info(f"Computing group statistics for partition '{part}' of {joined_dataset} dataset...")
        stats_df_per_n_bins = {}
        for n_bins in N_BINS_LIST:
            stats_df = create_prediction_group_statistics_strict_bins(
                config=config,
                predictions_df=predictions_df,
                target_df=target_df,
                prediction_col=config.mdl_task.prediction_col,
                target_col=config.mdl_task.target_col,
                joined_dataset=joined_dataset,
                n_bins=n_bins,
            )
            stats_df_per_n_bins[n_bins] = stats_df
            logger.info(
                f"Table of statistics for {n_bins} groups from partition: {part} in dataset: {joined_dataset} has been generated and logged.")
    else:
        stats_df_per_n_bins = None

    logger.info(f"Computing the metrics for partition '{part}' in {joined_dataset} dataset...")
    for metric_enum in config.mdl_task.evaluation_metrics:
        metric = get_metric_from_enum(config, metric_enum, pred_col=config.mdl_task.prediction_col)
        try:
            if keys is not None:
                predictions_df = predictions_df.loc[keys, :]
                target_df = target_df.loc[keys, :]
            score = metric.eval(target_df, predictions_df)
        except Exception as e:
            logger.error(f"Error while evaluating the metric {metric.get_short_name()} for partition '{part}': {e}")
            logger.warning(f"Setting the score to NaN for partition '{part}'.")
            score = np.nan
        metric_name = metric.get_short_name()
        if prefix is not None:
            metric_name = f"{prefix}_{metric_name}"
        scores[(metric_enum, metric_name)] = score
    if log_metrics_to_console:
        logger.info(f"Evaluated the predictions for partition '{part}' in {joined_dataset} dataset:\n" +
                    "\n".join(f"    - {name}: {score}" for (_, name), score in scores.items()))
    if log_metrics_to_mlflow:
        logger.info(f"Logging the metrics for partition '{part}' in {joined_dataset} dataset to MLFlow...")
        for (_, metric_name), score in scores.items():
            mlflow.log_metric(metric_name, score)
        logger.info(f"Logged the metrics for partition '{part}' in {joined_dataset} dataset to MLFlow.")
    return scores, stats_df_per_n_bins


def evaluate_predictions(config: Config, predictions_df: Dict[str, pd.DataFrame],
                         target_df: Dict[str, pd.DataFrame], dataset: str,
                         prefix: str = None,
                         log_metrics_to_mlflow: bool = True,
                         save_metrics_table: bool = True,
                         compute_group_stats: bool = True,
                         keys: Dict[str, pd.Index] = None) -> Tuple[
    Dict[str, List[Tuple[MetricEnum, str, float]]], pd.DataFrame]:
    """
    Evaluates the predictions for all partitions of the dataset.
    Args:
        config (Config): Configuration object containing settings and parameters.
        predictions_df (Dict[str, pd.DataFrame]): Dictionary of predictions for each partition.
        target_df (Dict[str, pd.DataFrame]): Dictionary of targets for each partition.
        dataset (str): Name of the dataset (e.g., "train" or "test").
        prefix (str, optional): Prefix for the dataset ('pure' or None). Defaults to None.
        log_metrics_to_mlflow (bool, optional): Whether to log metrics to MLflow. Defaults to True.
        save_metrics_table (bool, optional): Whether to save metrics table in MLflow. Defaults to True.
        compute_group_stats (bool, optional): Whether to compute group statistics for the predictions. Defaults to True.
        keys (Dict[str, pd.Index], optional): Dictionary of keys for each partition. Defaults to None.

    Returns:
        Tuple[Dict[str, List[Tuple[MetricEnum, str, float]]], pd.DataFrame]: A tuple containing:
            - scores_by_part: Dictionary of scores for each partition.
            - scores_df: DataFrame containing the scores for each metric and partition.
    """
    scores_by_part = {}
    scores_by_names = {}
    stats_df_by_n_bins = {}
    joined_dataset = f"{dataset}_{prefix}" if prefix is not None else dataset
    logger.info(f"Evaluating the predictions for {joined_dataset} dataset...")
    for part in predictions_df.keys():
        predictions_part_df = get_partition(predictions_df, part)
        target_part_df = get_partition(target_df, part)
        keys_part = get_partition(keys, part) if keys is not None else None
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            scores, stats_df_per_n_bins = evaluate_predictions_part(config, predictions_part_df, target_part_df,
                                                                    dataset, prefix, part, log_metrics_to_mlflow,
                                                                    compute_group_stats=compute_group_stats,
                                                                    keys=keys_part, log_metrics_to_console=False)
            scores_by_part[part] = scores
            if compute_group_stats and stats_df_per_n_bins is not None:
                for n_bins, stats_df in stats_df_per_n_bins.items():
                    if n_bins not in stats_df_by_n_bins:
                        stats_df_by_n_bins[n_bins] = []
                    stats_df_by_n_bins[n_bins].append(stats_df)
                    
        for (metric_enum, metric_name), score in scores_by_part[part].items():
            if metric_name not in scores_by_names:
                scores_by_names[metric_name] = {}
            scores_by_names[metric_name][part] = score
    # Create a DataFrame from scores_by_names
    scores_df = pd.DataFrame(scores_by_names).T
    scores_df.index.name = "metric"
    scores_df.columns.name = "part"
    scores_df = scores_df[sorted(scores_df.columns)]
    if save_metrics_table:
        save_metrics_table_in_mlflow(scores_df, dataset, prefix)
    # Average the metrics over the cross-validation folds
    if config.data.cv_enabled:
        mean_df = scores_df.mean(axis=1)
        std_df = scores_df.std(axis=1)
        metrics_cv_stats_df = pd.DataFrame({"cv_mean": mean_df, "cv_std": std_df})
        if save_metrics_table:
            save_metrics_cv_stats_in_mlflow(metrics_cv_stats_df, dataset, prefix)
        if log_metrics_to_mlflow:
            for stat in metrics_cv_stats_df.columns:
                for metric_name, score in metrics_cv_stats_df[stat].items():
                    # metric_name = f"{prefix}_{metric_name}" if prefix is not None else metric_name
                    mlflow.log_metric(f"{metric_name}_{stat}", score)
    elif log_metrics_to_mlflow:
        for metric_name, score in scores_df.iloc[:, 0].items():
            # metric_name = f"{prefix}_{metric_name}" if prefix is not None else metric_name
            mlflow.log_metric(metric_name, score)
    scores_table = scores_df.T.sort_index()
    scores_table.index.name = joined_dataset
    scores_table.columns = [name.replace(f"{joined_dataset}_", "") for name in scores_table.columns]
    mean_and_std_table = pd.DataFrame.from_records([scores_table.mean(), scores_table.std()],
                                                   index=pd.Index(["mean", "std"], name=joined_dataset))
    scores_table = scores_table.map(partial(round, ndigits=4))
    mean_and_std_table = mean_and_std_table.map(partial(round, ndigits=4))
    logger.info(f"Scores of {joined_dataset} predictions:\n" +
                tabulate(scores_table, headers="keys", tablefmt="psql", showindex=True) + "\n" +
                tabulate(mean_and_std_table, headers="keys", tablefmt="psql", showindex=True))
    # Create a DataFrame of averaged group statistics
    if compute_group_stats:
        logger.info(f"Computing group statistics for all partitions in {joined_dataset} dataset...")
        for n_bins, stats_dfs in stats_df_by_n_bins.items():
            logger.info(f"Computing group statistics for {n_bins} bins ({joined_dataset} dataset)...")
            create_average_prediction_group_statistics(
                config=config,
                stats_dfs=stats_dfs,
                joined_dataset=joined_dataset,
                n_bins=n_bins,
            )
            logger.info(f"Table of statistics for {n_bins} groups ({joined_dataset} dataset) has been generated and logged.")
    return scores_by_part, scores_df
