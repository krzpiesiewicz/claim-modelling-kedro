import logging
from typing import Callable, List, Sequence
from pandera.typing import Series
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config


logger = logging.getLogger(__name__)

def is_target_gt_zero(target: Series) -> Series[bool]:
    return target > 0


def shortened_seq_to_string(seq: Sequence) -> str:
    def val_to_str(val):
        return f"{val:.3f}"

    lst = [val_to_str(elem) for elem in seq]
    if len(lst) <= 10:
        return ", ".join(lst)
    else:
        return ", ".join(lst[:5]) + " ... " + ", ".join(lst[-5:])


def sample(
    config: Config,
    train_idx: pd.Index,
    target_df: pd.DataFrame,
    is_event: Callable[[Series], Series[bool]] = is_target_gt_zero
) -> pd.DataFrame:
    train_size = len(train_idx)
    target_df = target_df.loc[train_idx, :]
    lower_outliers_idx = target_df[target_df[config.mdl_task.target_col] < config.smpl.lower_bound].index
    upper_outliers_idx = target_df[target_df[config.mdl_task.target_col] > config.smpl.upper_bound].index
    lower_outliers = target_df.loc[lower_outliers_idx, config.mdl_task.target_col].sort_values().values
    upper_outliers = target_df.loc[upper_outliers_idx, config.mdl_task.target_col].sort_values().values
    logger.info(f"""Removing outliers from train dataset before sampling...
    - {len(lower_outliers_idx)} targets < {config.smpl.lower_bound}: {shortened_seq_to_string(lower_outliers)}
    - {len(upper_outliers_idx)} targets > {config.smpl.upper_bound}: {shortened_seq_to_string(upper_outliers)}
    """)
    events_idx = target_df[is_event(target_df[config.mdl_task.target_col])].index
    n_events = len(events_idx)
    n_events_in_sample = round(config.smpl.n_obs * config.smpl.target_ratio)
    n_non_events_in_sample = config.smpl.n_obs - n_events_in_sample
    logger.info(f"""Sampling from train dataset:
    - target: {config.mdl_info.target}
    - events: {n_events} ({round(100 * n_events / train_size,1)}%)
    - all observations: {train_size}""")
    logger.info(f"""Target sample:
    - sample size: {config.smpl.n_obs}
    - events to samples: {n_events_in_sample}
    - nonevents to sample: {n_non_events_in_sample}""")
    if train_size < config.smpl.n_obs:
        raise Exception(f"The size of train dataset {train_size} (train_idx) is less than the size of the sample "
                        f"{config.smpl.n_obs} (config.smpl.n_obs).")
    if n_events < n_events_in_sample:
        raise Exception(f"The number of events {n_events} in train dataset is less than "
                        f"{round(100 * config.smpl.target_ratio, 1)}% "
                        f"(config.smpl.target_ratio) of the sample size {config.smpl.n_obs}, which equals "
                        f"{n_events_in_sample}.")
    sample_events_idx = target_df.loc[events_idx, :].sample(
        n=n_events_in_sample,
        random_state=config.smpl.random_seed).index
    non_events_idx = train_idx.difference(events_idx)
    sample_non_events_idx = target_df.loc[non_events_idx, :].sample(
        n=n_non_events_in_sample,
        random_state=config.smpl.random_seed).index
    sample_idx = sample_events_idx.union(sample_non_events_idx)
    sample_df = target_df.loc[sample_idx, :]
    logger.info(f"""Done sampling. The sample::
    - sampled events: {is_event(sample_df).sum().iloc[0]}
    - all sampled observations: {len(sample_df)}""")
    sampled_policies = pd.Series(sample_idx)
    return sampled_policies
