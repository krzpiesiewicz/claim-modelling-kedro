import logging
from typing import Callable, Tuple, Dict

import mlflow
import numpy as np
import pandas as pd
from pandera.typing import Series

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p05_sampling.utils import sample_with_no_condition, \
    sample_with_target_ratio, get_all_samples, handle_outliers, get_actual_target_ratio, return_none
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys
from claim_modelling_kedro.pipelines.utils.datasets import get_mlflow_run_id_for_partition, get_partition

logger = logging.getLogger(__name__)


def get_sample_keys(
        config: Config,
        trg_df_handled_outliers: pd.DataFrame,
        is_event: Callable[[Series], Series[bool]]
) -> pd.Index:
    if config.smpl.target_ratio is not None and is_event is not return_none:
        # Sample with target ratio condition
        sample_keys = sample_with_target_ratio(config=config,
                                               target_df=trg_df_handled_outliers,
                                               train_keys=trg_df_handled_outliers.index,
                                               is_event=is_event)
    elif config.smpl.n_obs is not None:
        # Sample without any condition
        sample_keys = sample_with_no_condition(config=config,
                                               target_df=trg_df_handled_outliers,
                                               train_keys=trg_df_handled_outliers.index)
    else:  # No sampling
        sample_keys = get_all_samples(config=config,
                                      target_df=trg_df_handled_outliers,
                                      train_keys=trg_df_handled_outliers.index)
    return sample_keys


def sample_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        train_keys: pd.Index,
        calib_keys: pd.Index,
        is_event: Callable[[Series], Series[bool]]
) -> Tuple[Dict[str, pd.Index], pd.DataFrame, pd.DataFrame, float]:
    # Handle outliers in the train dataset
    train_trg_df_handled_outliers = handle_outliers(config=config, train_keys=train_keys, target_df=target_df)
    sample_keys = get_sample_keys(config=config,
                                  trg_df_handled_outliers=train_trg_df_handled_outliers,
                                  is_event=is_event)
    # If set, use also the calibration data for sampling
    if config.smpl.use_calib_data or config.smpl.use_calib_data_for_validation:
        logger.info(f"Sampling also from the calibration data for:"
                    "\n  - creating the training sample" if config.smpl.use_calib_data_for_validation else ""
                    "\n  - creating the validation sample" if config.smpl.use_calib_data_for_validation else "")
        calib_trg_df_handled_outliers = handle_outliers(config=config, train_keys=calib_keys, target_df=target_df)
        sample_calib_keys = get_sample_keys(config=config,
                                      trg_df_handled_outliers=calib_trg_df_handled_outliers,
                                      is_event=is_event)
        sample_keys = sample_keys.union(sample_calib_keys)
    sample_features_df = features_df.loc[sample_keys, :]
    sample_target_df = target_df.loc[sample_keys, :]
    actual_target_ratio = get_actual_target_ratio(config=config, sample_trg_df=sample_target_df, is_event=is_event)
    return sample_keys, sample_features_df, sample_target_df, actual_target_ratio


def sample(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        train_keys: Dict[str, pd.Index],
        calib_keys: Dict[str, pd.Index],
        is_event: Callable[[Series], Series[bool]]
) -> Tuple[Dict[str, pd.Index], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    logger.info(f"Sampling from the train dataset...")
    # assert that train_keys, calib_keys have the same partitions names
    logger.info(f"train_keys.keys(): {train_keys.keys()}")
    assert train_keys.keys() == calib_keys.keys()
    partitions_keys = train_keys.keys()
    # sample independently in each partition
    sample_keys = {}
    sample_features_df = {}
    sample_target_df = {}
    actual_target_ratios = []
    for part in partitions_keys:
        logger.info(f"Sampling from partition '{part}' of the train dataset...")
        train_keys_part = train_keys[part]()
        calib_keys_part = calib_keys[part]()
        logger.debug(f"type(train_keys_part): {type(train_keys_part)}")
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.debug(f"mlflow_subrun_id: {mlflow_subrun_id}")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            sample_keys_part, sample_features_part_df, sample_target_part_df, actual_target_ratio = sample_part(
                config, features_df, target_df, train_keys_part, calib_keys_part, is_event)
        sample_keys[part] = sample_keys_part
        sample_features_df[part] = sample_features_part_df
        sample_target_df[part] = sample_target_part_df
        if actual_target_ratio is not None:
            actual_target_ratios.append(actual_target_ratio)
    if len(actual_target_ratios) > 0:
        mlflow.log_metric("actual_target_ratio", np.mean(actual_target_ratios))
    return sample_keys, sample_features_df, sample_target_df


def train_val_split(
        config: Config,
        sample_keys: Dict[str, pd.Index],
        calib_keys: Dict[str, pd.Index],
        sample_target_df: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.Index], Dict[str, pd.DataFrame]]:
    logger.info(f"Splitting the sample into train and validation datasets...")
    partitions_keys = sample_keys.keys()
    sample_train_keys = {}
    sample_val_keys = {}
    for part in partitions_keys:
        logger.info(f"Splitting partition '{part}' of the sample into train and validation datasets...")
        if config.smpl.use_calib_data_for_validation:
            sample_keys_part = get_partition(sample_keys, part)
            calib_keys_part = get_partition(calib_keys, part)
            sample_train_keys[part] = sample_keys_part.difference(calib_keys_part)
            sample_val_keys[part] = sample_keys_part.intersection(calib_keys_part)
        elif config.smpl.split_train_val_enabled:
            sample_keys_part = get_partition(sample_keys, part)
            sample_target_df_part = get_partition(sample_target_df, part)
            mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
            with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
                sample_train_keys_part, sample_val_keys_part = get_stratified_train_test_split_keys(
                    target_df=sample_target_df_part.loc[sample_keys_part,:],
                    stratify_target_col=config.mdl_task.target_col,
                    test_size=config.smpl.split_val_size,
                    random_seed=config.smpl.split_val_random_seed
                )
                sample_train_keys[part] = sample_train_keys_part
                sample_val_keys[part] = sample_val_keys_part
        else:
            # If no validation split is needed, use the whole sample as train
            sample_train_keys = sample_keys
            sample_val_keys[part] = pd.Index([])
    return sample_train_keys, sample_val_keys
