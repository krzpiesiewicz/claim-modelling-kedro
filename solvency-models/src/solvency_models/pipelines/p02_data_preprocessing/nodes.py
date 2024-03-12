import logging
from typing import Tuple

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def split_features_and_targets(
        config: Config,
        raw_features_and_claims_numbers: pd.DataFrame,
        raw_severities: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting raw data files into: features, freqs and severities...")
    freqs = pd.DataFrame()
    freqs[config.data.policy_id_col] = pd.Series(raw_features_and_claims_numbers.IDpol, dtype=int)
    freqs[config.data.claims_freq_target_col] = raw_features_and_claims_numbers["ClaimNb"] / \
                                                raw_features_and_claims_numbers[
                                                         "Exposure"]
    features = raw_features_and_claims_numbers.drop(["ClaimNb", "Exposure"], axis=1)
    features[config.data.policy_id_col] = pd.Series(features[config.data.policy_id_col], dtype=int)
    ### TODO nie zgadzają się indeksy z polisami
    assert 1 == 0
    severities = pd.DataFrame()
    severities[config.data.policy_id_col] = pd.Series(raw_severities.IDpol, dtype=int)
    severities[config.data.claims_severity_target_col] = raw_severities["ClaimAmount"]
    logger.info("...splitting done.")
    return features, freqs, severities


def split_train_calib_backtest(
        config: Config,
        freqs: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    logger.info("Splitting policies into: train, calib and backtest datasets...")

    freqs = freqs.set_index(config.data.policy_id_col)

    from sklearn.model_selection import train_test_split

    rest, backtest_policies = train_test_split(
        freqs[config.data.claims_freq_target_col] > 0,
        test_size=config.data.backtest_size,
        random_state=config.data.split_random_seed,
    )
    backtest_policies = pd.Series(backtest_policies.index)
    train_policies, calib_policies = train_test_split(
        freqs.loc[rest.index, config.data.claims_freq_target_col] > 0,
        test_size=config.data.calib_size,
        random_state=config.data.split_random_seed,
    )
    train_policies = pd.Series(train_policies.index)
    calib_policies = pd.Series(calib_policies.index)
    all_size = freqs.size
    logger.info(f"""...split into:
    - train: {train_policies.size} ({round(100 * train_policies.size / all_size, 1)}%)
    - calib: {calib_policies.size} ({round(100 * calib_policies.size / all_size, 1)}%)
    - backtest: {backtest_policies.size} ({round(100 * backtest_policies.size / all_size, 1)}%)""")

    return train_policies, calib_policies, backtest_policies
