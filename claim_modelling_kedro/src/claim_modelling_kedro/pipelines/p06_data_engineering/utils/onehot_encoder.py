import logging
from typing import Dict

import mlflow
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.utils import assert_categorical_features
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger

logger = logging.getLogger(__name__)

# Define the artifact path for one-hot encoder
_ohe_artifact_path = "model_de/ohe"
_reference_categories_filename = "reference_categories.yml"


import numpy as np

import numpy as np

def custom_one_hot_encode(
    features_df: pd.DataFrame,
    ohe: OneHotEncoder,
    config: Config,
    reference_categories: Dict[str, str]
) -> pd.DataFrame:
    features_df_clean = features_df.copy()
    for col in features_df_clean.columns:
        unique_vals = features_df_clean[col].dropna().unique()
        if len(unique_vals) == 0:
            continue

        drop_cat = None
        # Drop reference category if enabled and present
        if getattr(config.de, "ohe_drop_reference_cat", False) and col in reference_categories:
            drop_cat = reference_categories[col]
            logger.info(f"Column {col} with reference column {reference_categories[col]} has values:\n" + features_df_clean[col].value_counts().to_string())
        # Drop first if enabled and no reference
        elif getattr(config.de, "ohe_drop_first", False):
            sorted_vals = sorted(unique_vals)
            drop_cat = sorted_vals[0]
        # Drop one for binary if enabled and not dropped above
        elif getattr(config.de, "ohe_drop_binary", False) and len(unique_vals) == 2:
            sorted_vals = sorted(unique_vals)
            drop_cat = sorted_vals[0]

        if drop_cat is not None:
            features_df_clean.loc[features_df_clean[col] == drop_cat, col] = np.nan

    ohe.set_params(drop=None)
    ohe_features = ohe.fit_transform(features_df_clean)
    columns = ohe.get_feature_names_out(features_df_clean.columns)
    ohe_df = pd.DataFrame(ohe_features.toarray(), columns=columns, index=features_df_clean.index)

    # Drop columns that end with '_None' or '_nan'
    ohe_df = ohe_df[[col for col in ohe_df.columns if not (col.endswith('_None') or col.endswith('_nan'))]]

    return ohe_df


def one_hot_encode_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                   mlflow_run_id: str = None) -> pd.DataFrame:
    assert config.de.is_ohe_enabled, "One-hot encoder is not enabled"
    assert_categorical_features(features_df)
    logger.info("One-hot encoding features by the MLFlow one hot encoder model...")
    # Load the one hot encoder model from the MLflow model registry
    ohe = MLFlowModelLoader("one-hot encoder model").load_model(path=_ohe_artifact_path, run_id=mlflow_run_id)
    # Load the reference categories from MLflow
    reference_categories_path = mlflow.artifacts.download_artifacts(
        artifact_path=f"{_ohe_artifact_path}/{_reference_categories_filename}",
        run_id=mlflow_run_id
    )
    with open(reference_categories_path) as f:
        reference_categories = yaml.safe_load(f)
    result = custom_one_hot_encode(features_df, ohe, config, reference_categories)
    logger.info("One-hot encoded the features.")
    return result


def fit_transform_one_hot_encoder(config: Config, features_df: pd.DataFrame, reference_categories: Dict[str, str]) -> pd.DataFrame:
    assert config.de.is_ohe_enabled, "One-hot encoder is not enabled"
    assert_categorical_features(features_df)
    logger.info("Fitting a one hot encoder model...")
    ohe = OneHotEncoder(handle_unknown='ignore', drop=None, min_frequency=config.de.ohe_min_frequency,
                        max_categories=config.de.ohe_max_categories)
    ohe_features = ohe.fit_transform(features_df)
    logger.info("One hot encoded the features.")
    # Save the one hot encoder model to MLFlow registry
    MLFlowModelLogger(ohe, "one-hot encoder model").log_model(_ohe_artifact_path)
    # Save the reference categories to MLFlow
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tempdir:
        ref_cat_path = os.path.join(tempdir, _reference_categories_filename)
        with open(ref_cat_path, "w") as f:
            yaml.safe_dump(reference_categories, f)
        mlflow.log_artifact(ref_cat_path, artifact_path=_ohe_artifact_path)
    return custom_one_hot_encode(features_df, ohe, config, reference_categories)

