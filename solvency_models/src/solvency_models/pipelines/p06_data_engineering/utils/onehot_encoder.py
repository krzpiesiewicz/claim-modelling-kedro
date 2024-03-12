import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils.utils import assert_categorical_features
from solvency_models.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger

logger = logging.getLogger(__name__)

# Define the artifact path for one-hot encoder
_ohe_artifact_path = "model_de/ohe"


def _one_hot_encode(features_df: pd.DataFrame, ohe: OneHotEncoder,
                    ohe_features: pd.DataFrame = None) -> pd.DataFrame:
    if ohe_features is None:
        ohe_features = ohe.transform(features_df)
    ohe_features_df = pd.DataFrame(ohe_features.toarray(), columns=ohe.get_feature_names_out(features_df.columns),
                                   index=features_df.index)
    logger.info(f"{len(features_df.columns)} categorical features "
                f"were one-hot encoded to {len(ohe_features_df.columns)} columns.")
    return ohe_features_df


def one_hot_encode_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                   mlflow_run_id: str = None) -> pd.DataFrame:
    assert config.de.is_ohe_enabled, "One-hot encoder is not enabled"
    assert_categorical_features(features_df)
    logger.info("One-hot encoding features by the MLFlow one hot encoder model...")
    # Load the one hot encoder model from the MLflow model registry
    ohe = MLFlowModelLoader("one-hot encoder model").load_model(path=_ohe_artifact_path, run_id=mlflow_run_id)
    logger.info("One-hot encoded the features.")
    return _one_hot_encode(features_df, ohe=ohe)


def fit_transform_one_hot_encoder(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    assert config.de.is_ohe_enabled, "One-hot encoder is not enabled"
    assert_categorical_features(features_df)
    logger.info("Fitting a one hot encoder model...")
    ohe = OneHotEncoder(handle_unknown='ignore', drop=config.de.ohe_drop, min_frequency=config.de.ohe_min_frequency,
                        max_categories=config.de.ohe_max_categories)
    ohe_features = ohe.fit_transform(features_df)
    logger.info("One hot encoded the features.")
    # Save the one hot encoder model to MLFlow registry
    MLFlowModelLogger(ohe, "one-hot encoder model").log_model(_ohe_artifact_path)
    return _one_hot_encode(features_df, ohe=ohe, ohe_features=ohe_features)
