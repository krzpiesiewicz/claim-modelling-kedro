import logging
from typing import Tuple

import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config

logger = logging.getLogger(__name__)


def preprocess_data(config: Config, raw_features_and_claims_numbers_df: pd.DataFrame,
                    raw_severities_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filtering erroneous policies...")
    # Filter out policies with no severity data
    raw_joined_df = raw_features_and_claims_numbers_df.merge(raw_severities_df, on=config.data.raw_policy_id_col,
                                                             how="left")
    # Compute the number of claims per year based on severity data
    aggregated_severities_df = raw_joined_df.groupby(config.data.raw_policy_id_col)[
        config.data.raw_claims_amount_col].agg(
        claim_sum='sum',
        claim_count='count'
    )
    # Join the aggregated severities to the raw data
    raw_joined_df = raw_features_and_claims_numbers_df.merge(aggregated_severities_df,
                                                             on=config.data.raw_policy_id_col, how="inner")
    # # Filter out policies where the claim_count does not match the number of claims in the frequency data
    # raw_filtered_joined_df = raw_joined_df[
    #     raw_joined_df.claim_count == raw_joined_df[config.data.raw_claims_number_col]]
    raw_filtered_joined_df = raw_joined_df
    # Filter out policies with claims amount greater than 4
    logger.info(f"Filtering out policies with claims amount greater than 4...")
    raw_filtered_joined_df = raw_filtered_joined_df[raw_filtered_joined_df["claim_count"] <= 4]
    # Set exposures greater than 1 to 1
    logger.info(f"Setting exposures greater than 1 to 1...")
    raw_filtered_joined_df.loc[
        raw_filtered_joined_df[config.data.raw_exposure_col] > 1, config.data.raw_exposure_col] = 1
    # Convert columns to the correct types and change names
    logger.info(f"Converting columns to the correct types and changing names...")
    filtered_joined_df = raw_filtered_joined_df.drop([
        config.data.raw_policy_id_col,
        config.data.raw_claims_number_col,
    ], axis=1)
    filtered_joined_df[config.data.policy_id_col] = raw_filtered_joined_df[config.data.raw_policy_id_col].astype(int)
    filtered_joined_df.rename(columns={
        config.data.raw_exposure_col: config.data.policy_exposure_col,
        "claim_count": config.data.claims_number_target_col,
        "claim_sum": config.data.claims_total_amount_target_col,
    }, inplace=True)
    # Add target columns adjusted to whole year
    logger.info(f"Adding target columns adjusted to whole year...")
    filtered_joined_df[config.data.claims_freq_target_col] = (
            filtered_joined_df[config.data.claims_number_target_col] /
            filtered_joined_df[config.data.policy_exposure_col])
    filtered_joined_df[config.data.claims_pure_premium_target_col] = (
            filtered_joined_df[config.data.claims_total_amount_target_col] /
            filtered_joined_df[config.data.policy_exposure_col])
    # Add average claim amount target column
    filtered_joined_df[config.data.claims_avg_amount_target_col] = (
            filtered_joined_df[config.data.claims_total_amount_target_col] /
            np.fmax(filtered_joined_df[config.data.claims_number_target_col], 1))
    # Set index to policy_id_col
    filtered_joined_df.set_index(config.data.policy_id_col, inplace=True)
    # Handle outliers
    from claim_modelling_kedro.pipelines.utils.outliers import handle_outliers
    filtered_joined_df, _, _ = handle_outliers(
        target_df=filtered_joined_df,
        target_col=config.data.stratify_target_col,
        outliers_conf=config.data.outliers,
        verbose=True
    )
    logger.info("All done. Created filtered joined dataframe.")
    return filtered_joined_df


def split_features_and_targets(
        config: Config,
        filtered_joined_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting filtered joined dataframe into: features and targets...")
    features_df = filtered_joined_df.drop(config.data.targets_cols, axis=1)
    targets_df = filtered_joined_df[config.data.targets_cols]
    logger.info("...splitting done.")
    return features_df, targets_df
