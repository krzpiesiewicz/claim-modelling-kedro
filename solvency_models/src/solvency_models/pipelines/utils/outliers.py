import logging
from typing import Tuple

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.utils.utils import shortened_seq_to_string

logger = logging.getLogger(__name__)


def remove_outliers(
        target_df: pd.DataFrame,
        target_col: str,
        keys: pd.Index = None,
        lower_bound: float = None,
        upper_bound: float = None,
        dataset_name: str = None
) -> Tuple[pd.Index, pd.DataFrame]:
    if dataset_name is None:
        dataset_name = ""
    else:
        dataset_name += " "
    if keys is None:
        keys = target_df.index
    else:
        target_df = target_df.loc[keys, :]
    msg = f"Removing outliers from the {dataset_name}dataset before sampling..."
    keys_without_outliers = keys # We will remove the outliers keys from keys_without_outliers:
    if lower_bound is not None:
        lower_outliers_idx = target_df[target_df[target_col] < lower_bound].index
        lower_outliers = target_df.loc[lower_outliers_idx, target_col].sort_values().values
        msg += (f"\n    - {len(lower_outliers_idx)} targets are less than {lower_bound}: "
                f"{shortened_seq_to_string(lower_outliers)}")
        logger.debug(f"lower_outliers: {lower_outliers}")
        keys_without_outliers = keys_without_outliers.difference(lower_outliers_idx)
    if upper_bound is not None:
        upper_outliers_idx = target_df[target_df[target_col] > upper_bound].index
        upper_outliers = target_df.loc[upper_outliers_idx, target_col].sort_values().values
        msg += (f"\n    - {len(upper_outliers_idx)} targets are greater than {upper_bound}: "
                f"{shortened_seq_to_string(upper_outliers)}")
        logger.debug(f"upper_outliers: {upper_outliers}")
        keys_without_outliers = keys_without_outliers.difference(upper_outliers_idx)
    logger.info(msg)
    trg_df_without_outliers = target_df.loc[keys_without_outliers, :]
    return keys_without_outliers, trg_df_without_outliers
