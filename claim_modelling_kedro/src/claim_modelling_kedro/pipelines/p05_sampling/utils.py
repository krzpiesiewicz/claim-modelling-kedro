import logging
from typing import Callable, Tuple

import mlflow
import pandas as pd
from pandera.typing import Series

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_sample_keys

logger = logging.getLogger(__name__)


def is_target_gt_zero(target: Series) -> Series[bool]:
    return target > 0


def return_none() -> None:
    return None


def remove_outliers(
        config: Config,
        train_keys: pd.Index,
        target_df: pd.DataFrame
) -> Tuple[pd.Index, pd.DataFrame]:
    from claim_modelling_kedro.pipelines.utils.outliers import remove_outliers as _remove_outliers

    train_keys_without_outliers, train_trg_df_without_outliers = _remove_outliers(
        keys=train_keys,
        target_df=target_df,
        target_col=config.mdl_task.target_col,
        lower_bound=config.smpl.lower_bound,
        upper_bound=config.smpl.upper_bound,
        dataset_name="sample"
    )
    return train_keys_without_outliers, train_trg_df_without_outliers


def assert_train_size_gte_given_config_sample_size(config: Config, train_trg_df: pd.DataFrame) -> pd.DataFrame:
    train_size = train_trg_df.shape[0]
    if train_size < config.smpl.n_obs:
        raise Exception(f"Cannot sample from the train dataset!!!\n"
                        f"The size of train dataset {train_size} (train_keys) is less than the size of the sample "
                        f"{config.smpl.n_obs} (config.smpl.n_obs).\n"
                        f"=> Consider changing the parameter sampling.n_obs to or lower than {train_size}.")
    return train_trg_df


def get_actual_target_ratio(config: Config, sample_trg_df: pd.DataFrame,
                            is_event: Callable[[Series], Series[bool]]) -> float:
    if is_event is return_none:
        return None
    sampled_events = is_event(sample_trg_df[config.mdl_task.target_col]).sum()
    sample_size = sample_trg_df.shape[0]
    actual_target_ratio = round(sampled_events / sample_size, 4)
    logger.info(f"""Actual target ratio in the sample: {actual_target_ratio}""")
    return actual_target_ratio


def get_all_samples(config: Config, target_df: pd.DataFrame, train_keys: pd.Index) -> pd.Index:
    train_size = len(train_keys)
    logger.info(f"""No sampling. All train observations except outliers will be used => sample size: ({train_size})""")
    return train_keys


def sample_with_no_condition(config: Config, target_df: pd.DataFrame, train_keys: pd.Index) -> pd.Index:
    train_trg_df = target_df.loc[train_keys, :]
    assert_train_size_gte_given_config_sample_size(config, train_trg_df)
    train_size = train_trg_df.shape[0]
    logger.info(f"""Sampling from the train dataset:
    - target: {config.mdl_info.target}
    - all observations: {train_size}
    - sample size: {config.smpl.n_obs}""")
    sample_keys = get_stratified_sample_keys(train_trg_df, config.data.stratify_target_col,
                                             size=config.smpl.n_obs, shuffle=True,
                                             random_seed=config.smpl.random_seed)
    sample_df = train_trg_df.loc[sample_keys, :]
    sample_size = sample_df.shape[0]
    train_size = train_trg_df.shape[0]
    logger.info(f"""Done sampling. The sample:
    - all sampled observations: {sample_size}  ({round(100 * sample_size / train_size, 1)}% of available observations)""")
    return sample_keys


def sample_with_target_ratio(config: Config, target_df: pd.DataFrame, train_keys: pd.Index,
                             is_event: Callable[[Series], Series[bool]]) -> Tuple[pd.Index, float]:
    train_trg_df = target_df.loc[train_keys, :]
    train_size = train_trg_df.shape[0]
    events_keys = train_trg_df[is_event(train_trg_df[config.mdl_task.target_col])].index
    n_events = events_keys.shape[0]
    if config.smpl.n_obs is None:
        n_obs = round(n_events / config.smpl.target_ratio)
        logger.info(f"""Parameter sampling.n_obs is not set. The sample size will be calculated based on the sampling.target_ratio and available events:
        - events: {n_events}
        - target_ratio: {config.smpl.target_ratio}
        => n_obs := {n_obs}.""")
        if n_obs > train_size:
            raise Exception(f"Cannot sample from the train dataset!!!\n"
                            f"The size of train dataset {train_size} (train_keys) is less than the size of the sample "
                            f"{n_obs} (calculated based on the sampling.target_ratio and available events).\n"
                            f"=> Consider changing the parameter sampling.target_ratio to or lower than {n_events / train_size}.")
    else:
        assert_train_size_gte_given_config_sample_size(config, train_trg_df)
        n_obs = config.smpl.n_obs
    n_events_in_sample = round(n_obs * config.smpl.target_ratio)
    n_non_events_in_sample = n_obs - n_events_in_sample
    logger.info(f"""Sampling from the train dataset:
    - target: {config.mdl_info.target}
    - events: {n_events} ({round(100 * n_events / train_size, 1)}%)
    - all observations: {train_size}""")
    logger.info(f"""Target sample:
    - sample size: {n_obs}
    - events to samples: {n_events_in_sample}
    - nonevents to sample: {n_non_events_in_sample}""")
    if n_events < n_events_in_sample:
        possible_target_ratio = n_events / n_obs
        message = (f"The number of events {n_events} in train dataset is less than "
                   f"{round(100 * config.smpl.target_ratio, 1)}% "
                   f"(parameter sampling.target_ratio) of the sample size {n_obs}, which equals "
                   f"{n_events_in_sample}.\n"
                   f"    1. Mind that {n_events} events would be {round(100 * possible_target_ratio, 1)}% "
                   f"of the {n_obs} sample.\n"
                   f"    2. Consider changing the parameter sampling.target_ratio to or lower than {possible_target_ratio}.")
        if config.smpl.allow_lower_ratio:
            logger.warning(message)
            logger.warning(f"Since sampling.allow_lower_ratio == {config.smpl.allow_lower_ratio},"
                           f"the sample will contain all events, that is {n_events} events which makes "
                           f"{round(100 * possible_target_ratio, 1)}% of the sample.")
            n_events_in_sample = n_events
        else:
            raise Exception(message)
    if n_events_in_sample == 0:
        sample_events_keys = pd.Index([], name=train_trg_df.index.name)
    else:
        sample_events_keys = get_stratified_sample_keys(train_trg_df.loc[events_keys, :],
                                                        config.data.stratify_target_col,
                                                        size=n_events_in_sample, shuffle=True,
                                                        random_seed=config.smpl.random_seed)
    non_events_keys = train_keys.difference(events_keys)
    if n_non_events_in_sample == 0:
        sample_non_events_keys = pd.Index([], name=train_trg_df.index.name)
    else:
        sample_non_events_keys = get_stratified_sample_keys(train_trg_df.loc[non_events_keys, :],
                                                            config.data.stratify_target_col,
                                                            size=n_non_events_in_sample, shuffle=True,
                                                            random_seed=config.smpl.random_seed)
    sample_keys = sample_events_keys.union(sample_non_events_keys)
    sample_df = train_trg_df.loc[sample_keys, :]
    sampled_events = is_event(sample_df[config.mdl_task.target_col]).sum()
    sample_size = sample_df.shape[0]
    actual_target_ratio = round(sampled_events / sample_size, 4)
    logger.info(f"""Done sampling:
    - sampled events: {sampled_events}  ({round(100 * sampled_events / n_events, 1)}% of available events)
    - all sampled observations: {sample_size}  ({round(100 * sample_size / train_size, 1)}% of available observations)""")
    rel_error_target_ratio = abs(config.smpl.target_ratio - actual_target_ratio) / config.smpl.target_ratio
    rel_error_threshold = 0.001
    if rel_error_target_ratio > rel_error_threshold:
        logger.warning(
            f"The actual target_ratio is {actual_target_ratio} that is {'less' if config.smpl.target_ratio - actual_target_ratio > 0 else 'greater'} than "
            f"{round(100 * (1 - rel_error_threshold), 1)}% of the theoretical target_ratio {config.smpl.target_ratio}.")
    mlflow.log_param("target_ratio_actual_value", actual_target_ratio)
    return sample_keys
