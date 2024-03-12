from typing import Tuple, List

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p07_data_science.hypertune import hypertune_transform
from solvency_models.pipelines.p07_data_science.model import fit_transform_predictive_model, \
    evaluate_predictions
from solvency_models.pipelines.p07_data_science.select import fit_transform_features_selector


def select_features(config: Config, transformed_sample_features_df: pd.DataFrame,
                    sample_target_df: pd.DataFrame) -> pd.DataFrame:
    if config.ds.fs_enabled:
        return fit_transform_features_selector(config, transformed_sample_features_df, sample_target_df)
    return transformed_sample_features_df


def fit_predictive_model(config: Config, selected_sample_features_df: pd.DataFrame,
                         sample_target_df: pd.DataFrame) -> pd.DataFrame:
    if config.ds.hopt_enabled:
        sample_predictions_df = hypertune_transform(config, selected_sample_features_df)
        evaluate_predictions(config, sample_predictions_df, sample_target_df, prefix="hopt", log_to_mlflow=True)
    else:
        sample_predictions_df = fit_transform_predictive_model(config, selected_sample_features_df, sample_target_df)
        evaluate_predictions(config, sample_predictions_df, sample_target_df, prefix="sample", log_to_mlflow=True)
    return sample_predictions_df
