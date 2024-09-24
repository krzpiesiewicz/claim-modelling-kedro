import logging
from typing import Tuple, Dict

import pandas as pd
from kedro.io import DataCatalog
from sklearn.model_selection import train_test_split

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.utils.stratified_cv_split import get_stratified_train_calib_test_cv
from solvency_models.pipelines.utils.stratified_split import get_stratified_train_calib_test_split_keys
from solvency_models.pipelines.utils.utils import remove_dataset
from solvency_models.hooks import get_catalog

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
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
    Dict[str, pd.Index], Dict[str, pd.Index], Dict[str, pd.Index]]:

    logger.info("Splitting policies into: train, calib and test datasets...")

    if config.data.cv_enabled:
        train_keys, calib_keys, test_keys = get_stratified_train_calib_test_cv(
            target_df,
            stratify_target_col=config.mdl_task.target_col,
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
            stratify_target_col=config.mdl_task.target_col,
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
