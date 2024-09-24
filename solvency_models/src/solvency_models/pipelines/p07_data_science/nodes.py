from typing import Dict, Any

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p07_data_science.hypertune import hypertune
from solvency_models.pipelines.p07_data_science.model import fit_transform_predictive_model, \
    evaluate_predictions
from solvency_models.pipelines.p07_data_science.select import fit_transform_features_selector
from solvency_models.pipelines.utils.utils import save_predictions_and_target_in_mlflow


def fit_features_selector(config: Config, transformed_sample_features_df: Dict[str, pd.DataFrame],
                          sample_target_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if config.ds.fs_enabled:
        return fit_transform_features_selector(config, transformed_sample_features_df, sample_target_df)
    return transformed_sample_features_df


def tune_hyper_parameters(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                          sample_target_df: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if config.ds.hopt_enabled:
        return hypertune(config, selected_sample_features_df, sample_target_df)


def fit_predictive_model(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                         sample_target_df: Dict[str, pd.DataFrame],
                         best_hparams: Dict[str, Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    # Fit the predictive model and transform the predictions
    sample_predictions_df = fit_transform_predictive_model(config, selected_sample_features_df, sample_target_df,
                                                           best_hparams)
    # Save the predictions and the target in MLFlow
    save_predictions_and_target_in_mlflow(sample_predictions_df, sample_target_df, dataset="sample")
    # Evaluate the predictions
    evaluate_predictions(config, sample_predictions_df, sample_target_df, dataset="sample", log_metrics_to_mlflow=True)
    return sample_predictions_df
