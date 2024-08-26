import logging
from typing import Tuple, Dict

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.utils.utils import get_partition

logger = logging.getLogger(__name__)


def remove_test_outliers_part(
        config: Config,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from solvency_models.pipelines.utils.outliers import remove_outliers as _remove_outliers

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
