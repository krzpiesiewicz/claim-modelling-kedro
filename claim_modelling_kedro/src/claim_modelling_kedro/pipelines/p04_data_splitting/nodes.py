import logging
from typing import Tuple, Dict

import mlflow
import numpy as np
from tabulate import tabulate

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight
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
        sample_weight = get_sample_weight(config, target_df)
        train_keys, calib_keys, test_keys = get_stratified_train_calib_test_cv(
            target_df,
            stratify_target_col=stratify_target_col,
            cv_folds=config.data.cv_folds,
            cv_parts_names=config.data.cv_parts_names,
            sample_weight=sample_weight,
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

    round_ndigits = 0 if target_df[stratify_target_col].mean() > 1000 else 2 # for rounding

    # Log the distribution of the target variable in each partition
    for dataset, keys_dct in zip(
            ["train", "calib", "test"],
            [train_keys, calib_keys, test_keys]
    ):
        top_n = 6
        stats_cols = ["n_obs", "mean", "weighted_mean", "min", "q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9", "q_0.95", "q_0.99", "q_0.995",
                "q_0.999"] + [f"top_{top_n - k}" for k in range(top_n)]
        table = pd.DataFrame({col: {} for col in ["dataset", "part"] + stats_cols}, index=range(len(keys_dct)))
        table[["dataset", "part"]] = table[["dataset", "part"]].astype(str)
        for row_idx, (part, keys) in enumerate(keys_dct.items()):
            weights = get_sample_weight(config, target_df.loc[keys,:])
            target_values = target_df.loc[keys, stratify_target_col]
            row = {
                "dataset": dataset,
                "part": part,
                "n_obs": len(target_values),
                "mean": target_values.mean(),
                "weighted_mean": np.average(target_values, weights=weights),
                "min": target_values.min(),
                "q_0.1": target_values.quantile(0.1),
                "q_0.25": target_values.quantile(0.25),
                "q_0.5": target_values.quantile(0.5),
                "q_0.75": target_values.quantile(0.75),
                "q_0.9": target_values.quantile(0.9),
                "q_0.95": target_values.quantile(0.95),
                "q_0.99": target_values.quantile(0.99),
                "q_0.995": target_values.quantile(0.995),
                "q_0.999": target_values.quantile(0.999),
            }
            largest_values = target_values.nlargest(top_n)
            for k in range(1, top_n + 1):
                row[f"top_{top_n + 1 - k}"] = round(largest_values.iloc[top_n - k], ndigits=round_ndigits)
            table.iloc[row_idx,:] = pd.Series(row)
        if round_ndigits == 0:
            table[stats_cols] = table[stats_cols].round().astype(int)
        csv_path = f"data/03_primary/{dataset}_distribution_table.csv"
        table.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="split")
        logger.info("\n" + tabulate(table, headers=table.columns, tablefmt="grid", showindex=False))

    return train_policies, calib_policies, test_policies, train_keys, calib_keys, test_keys
