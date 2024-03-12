import logging
from typing import Tuple

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def split_train_calib_test(
        config: Config,
        target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index, pd.Index, pd.Index]:
    logger.info("Splitting policies into: train, calib and test datasets...")
    from sklearn.model_selection import train_test_split

    rest, test_policies = train_test_split(
        target_df[config.data.claims_freq_target_col] > 0,
        test_size=config.data.test_size,
        random_state=config.data.split_random_seed,
    )
    if config.data.calib_size == 0:
        train_policies = rest
        calib_policies = pd.Series([], index=pd.Index([], name=rest.index.name))
    else:
        train_policies, calib_policies = train_test_split(
            target_df.loc[rest.index, config.data.claims_freq_target_col] > 0,
            test_size=config.data.calib_size / (1 - config.data.test_size),
            random_state=config.data.split_random_seed,
        )
    train_keys = train_policies.index
    calib_keys = calib_policies.index
    test_keys = test_policies.index
    test_policies = pd.Series(test_policies.index).to_frame()
    train_policies = pd.Series(train_policies.index).to_frame()
    calib_policies = pd.Series(calib_policies.index).to_frame()
    all_size = target_df.shape[0]
    logger.info(f"""...split into:
    - train: {train_policies.size} ({round(100 * train_policies.size / all_size, 1)}%)
    - calib: {calib_policies.size} ({round(100 * calib_policies.size / all_size, 1)}%)
    - test: {test_policies.size} ({round(100 * test_policies.size / all_size, 1)}%)""")
    logger.debug(f"""test_policies: {test_policies.head()}""")

    return train_policies, calib_policies, test_policies, train_keys, calib_keys, test_keys
