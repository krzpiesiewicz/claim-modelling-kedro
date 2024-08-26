import logging
from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from solvency_models.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def _split_train_calib(config: Config, train_calib_idx: pd.Index, target_df: pd.DataFrame) -> Tuple[
    pd.Series, pd.Series, pd.Index, pd.Index]:
    if config.data.calib_size == 0:
        train_policies = train_calib_idx
        calib_policies = pd.Series([], index=pd.Index([], name=target_df.index.name))
    else:
        train_policies, calib_policies = train_test_split(
            target_df.loc[train_calib_idx, config.data.claims_freq_target_col] > 0,
            test_size=config.data.calib_size / (1 - config.data.test_size),
            random_state=config.data.split_random_seed,
        )
    train_keys = train_policies.index
    calib_keys = calib_policies.index
    return train_policies, calib_policies, train_keys, calib_keys


def split_train_calib_test(
        config: Config,
        target_df: pd.DataFrame
        # ):
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Index], Dict[str, pd.Index],
    Dict[str, pd.Index]]:
    # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index, pd.Index, pd.Index]:
    logger.info("Splitting policies into: train, calib and test datasets...")

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

    one_part_key = config.data.one_part_key
    train_policies = {one_part_key: train_policies}
    calib_policies = {one_part_key: calib_policies}
    test_policies = {one_part_key: test_policies}

    train_keys = {one_part_key: train_keys}
    calib_keys = {one_part_key: calib_keys}
    test_keys = {one_part_key: test_keys}

    return train_policies, calib_policies, test_policies, train_keys, calib_keys, test_keys
