from typing import Dict, Tuple

import mlflow
import logging
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.pred_eval import evaluate_predictions
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition


logger = logging.getLogger(__name__)


def recompute_metrics(
        dummy_load_pred_and_trg_df: pd.DataFrame,
        config: Config,
        sample_train_predictions_df: Dict[str, pd.DataFrame],
        sample_train_target_df: Dict[str, pd.DataFrame],
        sample_valid_predictions_df: Dict[str, pd.DataFrame],
        sample_valid_target_df: Dict[str, pd.DataFrame],
        calib_predictions_df: Dict[str, pd.DataFrame],
        calib_target_df: Dict[str, pd.DataFrame],
        train_predictions_df: Dict[str, pd.DataFrame],
        train_target_df: Dict[str, pd.DataFrame],
        test_predictions_df: Dict[str, pd.DataFrame],
        test_target_df: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Reevaluates predictions and recomputes metrics for different datasets and partitions

    Args:
        config (Config): Configuration object containing pipeline settings.
        sample_train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample train predictions for each partition.
        sample_train_target_df (Dict[str, pd.DataFrame]): Dictionary of sample train targets for each partition.
        sample_valid_predictions_df (Dict[str, pd.DataFrame]): Dictionary of sample valid predictions for each partition.
        sample_valid_target_df (Dict[str, pd.DataFrame]): Dictionary of sample valid targets for each partition.
        calib_predictions_df (Dict[str, pd.DataFrame]): Dictionary of calibrated predictions for each partition.
        calib_target_df (Dict[str, pd.DataFrame]): Dictionary of calibrated targets for each partition.
        train_predictions_df (Dict[str, pd.DataFrame]): Dictionary of train predictions for each partition.
        train_target_df (Dict[str, pd.DataFrame]): Dictionary of train targets for each partition.
        test_predictions_df (Dict[str, pd.DataFrame]): Dictionary of test predictions for each partition.
        test_target_df (Dict[str, pd.DataFrame]): Dictionary of test targets for each partition.
    """
    # Iterate over train and test datasets
    for dataset, predictions_df, target_df in zip(
            ["sample_train", "sample_valid", "calib", "train", "test"],
            [sample_train_predictions_df, sample_valid_predictions_df, calib_predictions_df, train_predictions_df, test_predictions_df],
            [sample_train_target_df, sample_valid_target_df, calib_target_df, train_target_df, test_target_df]):
        if len(predictions_df) == 0:
            logger.warning(f"Dataset {dataset} is empty. Skipping...")
            continue
        if dataset.startswith("sample"):
            prefixes_and_columns = [("pure", config.mdl_task.prediction_col)]
        else:
            prefixes_and_columns = [(None, config.mdl_task.prediction_col)]
            if config.clb.enabled and not dataset.startswith("sample"):
                prefixes_and_columns.append(("pure", config.clb.pure_prediction_col))
        for prefix, prediction_col in prefixes_and_columns:
            # Evaluate the predictions
            evaluate_predictions(config, predictions_df, target_df, dataset=dataset, prefix=prefix,
                                 log_metrics_to_mlflow=True, save_metrics_table=True, compute_group_stats=True)

    dummy_recomp_metrics_df = pd.DataFrame({})
    return dummy_recomp_metrics_df
