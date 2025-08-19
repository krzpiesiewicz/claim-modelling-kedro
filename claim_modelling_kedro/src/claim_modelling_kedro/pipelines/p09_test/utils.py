import logging
from typing import Dict, Tuple

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils import transform_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.model import predict_by_mlflow_model, evaluate_predictions
from claim_modelling_kedro.pipelines.p07_data_science.select import select_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import \
    inverse_transform_predictions_by_mlflow_model
from claim_modelling_kedro.pipelines.p08_model_calibration.utils import calibrate_predictions_by_mlflow_model
from claim_modelling_kedro.pipelines.utils.datasets import get_partition
from claim_modelling_kedro.pipelines.utils.dataframes import save_predictions_and_target_in_mlflow

logger = logging.getLogger(__name__)


def handle_outliers_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from claim_modelling_kedro.pipelines.utils.outliers import handle_outliers as _handle_outliers

    # Handle outliers in the test dataset
    trg_df_handled_outliers, *_ = _handle_outliers(
        target_df=target_df,
        target_col=config.mdl_task.target_col,
        outliers_conf=config.test.outliers,
        dataset_name="test",
        verbose=True
    )
    features_df_handled_outliers = features_df.loc[trg_df_handled_outliers.index, :]
    return features_df_handled_outliers, trg_df_handled_outliers


def handle_outliers(
        config: Config,
        features_df: Dict[str, pd.DataFrame],
        target_df: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    features_df_handled_outliers = {}
    target_df_handled_outliers = {}
    logger.info("Handling outliers in the test dataset...")
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        target_part_df = get_partition(target_df, part)
        logger.info(f"Handling outliers in partition '{part}' of the test dataset...")
        features_df_handled_outliers[part], target_df_handled_outliers[part] = handle_outliers_part(
            config, features_part_df, target_part_df
        )
    logger.info("Handled the outliers in the test dataset.")
    return features_df_handled_outliers, target_df_handled_outliers


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

    features_wo_outliers, target_wo_outliers = handle_outliers(
        config=config,
        features_df=part_features_df,
        target_df=part_target_df
    )
    transformed_features = transform_features_by_mlflow_model(
        config, features_wo_outliers, mlflow_run_id=config.de.mlflow_run_id
    )
    if config.ds.fs.enabled:
        selected_features = select_features_by_mlflow_model(
            config, transformed_features, mlflow_run_id=config.ds.mlflow_run_id
        )
    else:
        selected_features = transformed_features
    pure_predictions = predict_by_mlflow_model(
        config, selected_features, mlflow_run_id=config.ds.mlflow_run_id
    )
    if config.ds.trg_trans.enabled:
        pure_predictions = inverse_transform_predictions_by_mlflow_model(config, pure_predictions,
                                                                         mlflow_run_id=config.ds.mlflow_run_id)
    if config.clb.enabled:
        calibrated_predictions = calibrate_predictions_by_mlflow_model(
            config, pure_predictions, mlflow_run_id=config.clb.mlflow_run_id
        )
    else:
        calibrated_predictions = pure_predictions

    save_predictions_and_target_in_mlflow(calibrated_predictions, part_target_df, dataset=dataset_name)
    evaluate_predictions(config, pure_predictions, part_target_df, dataset=dataset_name, prefix="pure",
                         log_metrics_to_mlflow=True, save_metrics_table=True)
    evaluate_predictions(config, calibrated_predictions, part_target_df, dataset=dataset_name,
                         log_metrics_to_mlflow=True, save_metrics_table=True)

    return calibrated_predictions, part_target_df
