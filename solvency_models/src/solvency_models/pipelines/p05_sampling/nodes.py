import logging
from typing import Callable, Tuple, Dict

import pandas as pd
from pandera.typing import Series

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p05_sampling.utils import sample_with_no_condition, \
    sample_with_target_ratio, get_all_samples, return_none, remove_outliers

logger = logging.getLogger(__name__)


def sample_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        train_keys: pd.Index,
        calib_keys: pd.Index,
        is_event: Callable[[Series], Series[bool]]
) -> Tuple[Dict[str, pd.Index], pd.DataFrame, pd.DataFrame]:
    # If set, use also the calibration data for sampling
    if config.smpl.use_calib_data:
        train_keys = train_keys.union(calib_keys)
        logger.info(f"Using also the calibration data for sampling.")
    # Remove outliers from the train dataset
    train_keys_without_outliers, train_trg_df_without_outliers = remove_outliers(config=config, train_keys=train_keys,
                                                                                 target_df=target_df)
    if config.smpl.n_obs is not None:
        # Sample from the train dataset
        if is_event is return_none:
            # Sample without any condition
            sample_keys = sample_with_no_condition(config=config,
                                                   target_df=train_trg_df_without_outliers,
                                                   train_keys=train_keys_without_outliers)
        else:
            # Sample with target ratio condition
            sample_keys = sample_with_target_ratio(config=config,
                                                   target_df=train_trg_df_without_outliers,
                                                   train_keys=train_keys_without_outliers,
                                                   is_event=is_event)
    else:  # No sampling
        sample_keys = get_all_samples(config=config,
                                      target_df=train_trg_df_without_outliers,
                                      train_keys=train_keys_without_outliers)

    sample_features_df = features_df.loc[sample_keys, :]
    sample_target_df = target_df.loc[sample_keys, :]
    return sample_keys, sample_features_df, sample_target_df


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
    assert train_keys.keys() == calib_keys.keys()
    partitions_keys = train_keys.keys()
    # sample independently in each partition
    sample_keys = {}
    sample_features_df = {}
    sample_target_df = {}
    for part in partitions_keys:
        logger.info(f"Sampling from partition '{part}' of the train dataset...")
        train_keys_part = train_keys[part]()
        calib_keys_part = calib_keys[part]()
        logger.debug(f"type(train_keys_part): {type(train_keys_part)}")
        sample_keys_part, sample_features_part_df, sample_target_part_df = sample_part(
            config, features_df, target_df, train_keys_part, calib_keys_part, is_event)
        sample_keys[part] = sample_keys_part
        sample_features_df[part] = sample_features_part_df
        sample_target_df[part] = sample_target_part_df
    return sample_keys, sample_features_df, sample_target_df
