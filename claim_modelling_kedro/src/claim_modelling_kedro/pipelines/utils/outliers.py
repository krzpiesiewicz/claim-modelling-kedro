import logging
from typing import Tuple

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.outliers_config import OutliersConfig, OutliersPolicy
from claim_modelling_kedro.pipelines.utils.utils import shortened_seq_to_string

logger = logging.getLogger(__name__)


def handle_outliers(
        target_df: pd.DataFrame,
        target_col: str,
        outliers_conf: OutliersConfig,
        keys: pd.Index = None,
        dataset_name: str = None,
        verbose: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if dataset_name is None:
        dataset_name = ""
    else:
        dataset_name += " "
    if keys is None:
        keys = target_df.index
    else:
        target_df = target_df.loc[keys, :]
    msg = f"Handling outliers from the {dataset_name}dataset..."
    if outliers_conf.lower_bound is None:
        lower_outliers_idx = pd.Index([])
        lower_outliers = None
    else:
        lower_outliers_idx = target_df[target_df[target_col] < outliers_conf.lower_bound].index
        lower_outliers = target_df.loc[lower_outliers_idx, target_col].sort_values()
        msg += (f"\n    - {len(lower_outliers_idx)} targets are less than {outliers_conf.lower_bound}: "
                f"{shortened_seq_to_string(lower_outliers.values)}")
        logger.debug(f"lower_outliers: {lower_outliers.values}")
    if outliers_conf.upper_bound is None:
        upper_outliers_idx = pd.Index([])
        upper_outliers = None
    else:
        upper_outliers_idx = target_df[target_df[target_col] > outliers_conf.upper_bound].index
        upper_outliers = target_df.loc[upper_outliers_idx, target_col].sort_values()
        msg += (f"\n    - {len(upper_outliers_idx)} targets are greater than {outliers_conf.upper_bound}: "
                f"{shortened_seq_to_string(upper_outliers.values)}")
        logger.debug(f"upper_outliers: {upper_outliers.values}")
    if verbose:
        logger.info(msg)
    match outliers_conf.policy:
        case OutliersPolicy.DROP:
            if verbose:
                logger.info(f"Dropping the outliers from the {dataset_name}dataset...")
            keys_handled_outliers = keys.difference(lower_outliers_idx).difference(upper_outliers_idx)
            trg_df_with_handled_outliers = target_df.loc[keys_handled_outliers, :]
        case OutliersPolicy.CLIP:
            if verbose:
                values_range = "(-inf" if outliers_conf.lower_bound is None else f"[{outliers_conf.lower_bound}"
                values_range += ", "
                values_range += "inf)" if outliers_conf.upper_bound is None else f"{outliers_conf.upper_bound}]"
                logger.info(f"Clipping the outliers from the {dataset_name}dataset to range {values_range}...")
            trg_df_with_handled_outliers = target_df.copy()
            trg_df_with_handled_outliers.loc[lower_outliers_idx, target_col] = outliers_conf.lower_bound
            trg_df_with_handled_outliers.loc[upper_outliers_idx, target_col] = outliers_conf.upper_bound
        case OutliersPolicy.KEEP:
            if verbose:
                logger.info(f"Keeping the outliers from the {dataset_name}dataset...")
            trg_df_with_handled_outliers = target_df
        case _:
            raise ValueError(f"Unknown outliers policy: {outliers_conf.policy}.")
    if verbose:
        logger.info(f"Handled the outliers from the {dataset_name}dataset.")
    logger.debug(f"""Statistics:
    - {len(trg_df_with_handled_outliers)} targets in the {dataset_name}dataset
    - minimum target: {trg_df_with_handled_outliers[target_col].min()}
    - maximum target: {trg_df_with_handled_outliers[target_col].max()}
    """)
    return trg_df_with_handled_outliers, upper_outliers, lower_outliers
