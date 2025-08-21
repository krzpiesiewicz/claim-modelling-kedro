from functools import partial
from typing import Dict, Tuple, List

import logging
import mlflow
import numpy as np
import pandas as pd
from tabulate import tabulate

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum
from claim_modelling_kedro.pipelines.utils.dataframes import save_metrics_table_in_mlflow, \
    save_metrics_cv_stats_in_mlflow
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition
from claim_modelling_kedro.pipelines.utils.metrics import get_metric_from_enum


logger = logging.getLogger(__name__)


def evaluate_predictions(config: Config, predictions_df: Dict[str, pd.DataFrame],
                         target_df: Dict[str, pd.DataFrame], dataset: str,
                         prefix: str = None,
                         log_metrics_to_mlflow: bool = True,
                         save_metrics_table: bool = True,
                         keys: Dict[str, pd.Index] = None) -> Tuple[
    Dict[str, List[Tuple[MetricEnum, str, float]]], pd.DataFrame]:
    scores_by_part = {}
    scores_by_names = {}
    logger.info("Evaluating the predictions...")
    joined_prefix = f"{dataset}_{prefix}" if prefix is not None else dataset
    for part in predictions_df.keys():
        predictions_part_df = get_partition(predictions_df, part)
        target_part_df = get_partition(target_df, part)
        keys_part = get_partition(keys, part) if keys is not None else None
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            scores_by_part[part] = evaluate_predictions_part(config, predictions_part_df, target_part_df,
                                                             joined_prefix, part, log_metrics_to_mlflow, keys=keys_part,
                                                             log_metrics_to_console=False)
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
    scores_table.index.name = joined_prefix
    scores_table.columns = [name.replace(f"{joined_prefix}_", "") for name in scores_table.columns]
    mean_and_std_table = pd.DataFrame.from_records([scores_table.mean(), scores_table.std()],
                                                   index=pd.Index(["mean", "std"], name=joined_prefix))
    scores_table = scores_table.map(partial(round, ndigits=4))
    mean_and_std_table = mean_and_std_table.map(partial(round, ndigits=4))
    logger.info(f"Scores of {joined_prefix} predictions:\n" +
                tabulate(scores_table, headers="keys", tablefmt="psql", showindex=True) + "\n" +
                tabulate(mean_and_std_table, headers="keys", tablefmt="psql", showindex=True))
    return scores_by_part, scores_df


def evaluate_predictions_part(config: Config, predictions_df: pd.DataFrame,
                              target_df: pd.DataFrame, prefix: str, part: str, log_metrics_to_mlflow: bool,
                              log_metrics_to_console: bool = True, keys: pd.Index = None) -> List[
    Tuple[MetricEnum, str, float]]:
    """
    prefix: str - default None, but can by any string like train, test, pure, cal etc.
    If prefix is not None, it will be added to the metric name separated by underscore.
    E.g., train_RMSE
    """
    logger.info(f"Evaluating the predictions for partition '{part}' of {prefix} dataset...")
    scores = {}
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
        logger.info(f"Evaluated the predictions for partition '{part}':\n" +
                    "\n".join(f"    - {name}: {score}" for (_, name), score in scores.items()))
    if log_metrics_to_mlflow:
        logger.info(f"Logging the metrics for partition '{part}' to MLFlow...")
        for (_, metric_name), score in scores.items():
            mlflow.log_metric(metric_name, score)
        logger.info(f"Logged the metrics for partition '{part}' to MLFlow.")
    return scores
