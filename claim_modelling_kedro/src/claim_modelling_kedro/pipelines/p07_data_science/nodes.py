from typing import Dict, Any

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.hypertune import hypertune
from claim_modelling_kedro.pipelines.p07_data_science.model import fit_transform_predictive_model, \
    evaluate_predictions
from claim_modelling_kedro.pipelines.p07_data_science.select import fit_transform_features_selector
from claim_modelling_kedro.pipelines.utils.utils import save_predictions_and_target_in_mlflow


def fit_features_selector(config: Config, transformed_sample_features_df: Dict[str, pd.DataFrame],
                          sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                          sample_val_keys: Dict[str, pd.Index]) -> Dict[str, pd.DataFrame]:
    if config.ds.fs_enabled:
        return fit_transform_features_selector(config, transformed_sample_features_df, sample_target_df, sample_train_keys, sample_val_keys)
    return transformed_sample_features_df


def tune_hyper_parameters(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                          sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                          sample_val_keys: Dict[str, pd.Index]) -> Dict[str, Dict[str, Any]]:
    if config.ds.hopt_enabled:
        return hypertune(config, selected_sample_features_df, sample_target_df, sample_train_keys, sample_val_keys)
    return {part: config.ds.model_const_hparams for part in selected_sample_features_df.keys()}


def fit_predictive_model(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
                         sample_target_df: Dict[str, pd.DataFrame], sample_train_keys: Dict[str, pd.Index],
                         sample_val_keys: Dict[str, pd.Index],
                         best_hparams: Dict[str, Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    # Fit the predictive model and transform the predictions
    sample_predictions_df = fit_transform_predictive_model(
        config,
        selected_sample_features_df,
        sample_target_df,
        sample_train_keys,
        sample_val_keys,
        best_hparams
    )
    # Save the predictions and the target in MLFlow
    save_predictions_and_target_in_mlflow(sample_predictions_df, sample_target_df, dataset="sample")
    # Evaluate the predictions
    evaluate_predictions(config, sample_predictions_df, sample_target_df, dataset="sample_train_pure",
                         log_metrics_to_mlflow=True, keys=sample_train_keys)
    evaluate_predictions(config, sample_predictions_df, sample_target_df, dataset="sample_valid_pure",
                         log_metrics_to_mlflow=True, keys=sample_val_keys)
    return sample_predictions_df
