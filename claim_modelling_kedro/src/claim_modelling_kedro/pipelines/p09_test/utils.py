import logging
from typing import Dict, Tuple

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.model import predict_by_mlflow_model, evaluate_predictions
from claim_modelling_kedro.pipelines.p07_data_science.select import select_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p08_model_calibration.utils import calibrate_predictions_by_mlflow_model
from claim_modelling_kedro.pipelines.utils.utils import get_partition, save_predictions_and_target_in_mlflow

logger = logging.getLogger(__name__)


def remove_test_outliers_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from claim_modelling_kedro.pipelines.utils.outliers import remove_outliers as _remove_outliers

    # Remove outliers from the test dataset
    logger.info("Removing outliers from the test dataset...")
    keys_without_outliers, trg_df_without_outliers = _remove_outliers(
        target_df=target_df,
        target_col=config.mdl_task.target_col,
        lower_bound=config.test.lower_bound,
        upper_bound=config.test.upper_bound,
        dataset_name="test"
    )
    features_df_without_outliers = features_df.loc[keys_without_outliers, :]
    logger.info("Removed the outliers.")
    return features_df_without_outliers, trg_df_without_outliers


def remove_test_outliers(
        config: Config,
        features_df: Dict[str, pd.DataFrame],
        target_df: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    features_df_without_outliers = {}
    target_df_without_outliers = {}
    logger.info("Removing outliers from the test dataset...")
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        target_part_df = get_partition(target_df, part)
        logger.info(f"Removing outliers from partition '{part}' of the test dataset...")
        features_df_without_outliers[part], target_df_without_outliers[part] = remove_test_outliers_part(
            config, features_part_df, target_part_df
        )
    logger.info("Removed the outliers.")
    return features_df_without_outliers, target_df_without_outliers


def run_eval_pipeline(
    config: Config,
    features_df: Dict[str, pd.DataFrame],
    target_df: Dict[str, pd.DataFrame],
    keys: Dict[str, pd.Index],
    dataset_name: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    logger.info(f"Prepare {dataset_name} dataset...")
    part_target_df = {}
    part_features_df = {}
    for part in keys.keys():
        part_keys = get_partition(keys, part)
        part_target_df[part] = target_df.loc[part_keys, :]
        part_features_df[part] = features_df.loc[part_keys, :]

    features_wo_outliers, target_wo_outliers = remove_test_outliers(
        config=config,
        features_df=part_features_df,
        target_df=part_target_df
    )
    transformed_features = transform_features_by_mlflow_model(
        config, features_wo_outliers, mlflow_run_id=config.de.mlflow_run_id
    )
    selected_features = select_features_by_mlflow_model(
        config, transformed_features, mlflow_run_id=config.ds.mlflow_run_id
    )
    pure_predictions = predict_by_mlflow_model(
        config, selected_features, mlflow_run_id=config.ds.mlflow_run_id
    )
    calibrated_predictions = calibrate_predictions_by_mlflow_model(
        config, pure_predictions, mlflow_run_id=config.clb.mlflow_run_id
    )

    save_predictions_and_target_in_mlflow(calibrated_predictions, part_target_df, dataset=dataset_name)
    evaluate_predictions(config, pure_predictions, part_target_df, dataset=dataset_name, prefix="pure",
                         log_metrics_to_mlflow=True, save_metrics_table=True)
    evaluate_predictions(config, calibrated_predictions, part_target_df, dataset=dataset_name,
                         log_metrics_to_mlflow=True, save_metrics_table=True)

    return calibrated_predictions, part_target_df
