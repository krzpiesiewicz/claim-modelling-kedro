import logging
from typing import Tuple

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def remove_test_outliers(
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
