import logging
from typing import Tuple, Dict

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.stratified_cv_split import get_stratified_train_calib_test_cv
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_calib_test_split_keys

logger = logging.getLogger(__name__)


def split_train_calib_test(
        config: Config,
        target_df: pd.DataFrame
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
    Dict[str, pd.Index], Dict[str, pd.Index], Dict[str, pd.Index]]:

    logger.info("Splitting policies into: train, calib and test datasets...")
    if config.data.stratify_target_col is not None:
        stratify_target_col = config.data.stratify_target_col
    else:
        logger.info("No stratification column provided. Using the target column for stratification.")
        stratify_target_col = config.mdl_task.target_col
    logger.info(f"Stratifying on column: {stratify_target_col}")

    if config.data.cv_enabled:
        train_keys, calib_keys, test_keys = get_stratified_train_calib_test_cv(
            target_df,
            stratify_target_col=stratify_target_col,
            cv_folds=config.data.cv_folds,
            cv_parts_names=config.data.cv_parts_names,
            shuffle=True,
            random_seed=config.data.split_random_seed,
            verbose=True,
        )
        train_policies = {}
        calib_policies = {}
        test_policies = {}
        for part in train_keys.keys():
            train_policies[part] = pd.Series(train_keys[part]).to_frame()
            calib_policies[part] = pd.Series(calib_keys[part]).to_frame()
            test_policies[part] = pd.Series(test_keys[part]).to_frame()

    else:
        train_keys, calib_keys, test_keys = get_stratified_train_calib_test_split_keys(
            target_df,
            stratify_target_col=stratify_target_col,
            test_size=config.data.test_size,
            calib_size=config.data.calib_size,
            shuffle=True,
            random_seed=config.data.split_random_seed,
            verbose=True,
        )
        one_part_key = config.data.one_part_key
        train_policies = {one_part_key: pd.Series(train_keys).to_frame()}
        calib_policies = {one_part_key: pd.Series(calib_keys).to_frame()}
        test_policies = {one_part_key: pd.Series(test_keys).to_frame()}
        train_keys = {one_part_key: train_keys}
        calib_keys = {one_part_key: calib_keys}
        test_keys = {one_part_key: test_keys}

    return train_policies, calib_policies, test_policies, train_keys, calib_keys, test_keys
