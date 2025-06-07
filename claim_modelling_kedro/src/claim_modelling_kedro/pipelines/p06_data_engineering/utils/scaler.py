import logging
from typing import Union

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.utils import assert_numerical_features
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger

logger = logging.getLogger(__name__)

# Define the artifact path for scaler
_scaler_artifact_path = "model_de/scaler"


def _scale_features(features_df: pd.DataFrame,
                    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler] = None,
                    scaled_features: pd.DataFrame = None) -> pd.DataFrame:
    assert scaler is not None or scaled_features is not None, "Either scaler or scaled_features should be provided"
    if scaled_features is None:
        scaled_features = scaler.transform(features_df)
    return pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)


def scale_features_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                   mlflow_run_id: str = None) -> pd.DataFrame:
    assert config.de.is_scaler_enabled, "Scaler is not enabled"
    assert_numerical_features(features_df)
    logger.info("Scaling features by the MLFlow scaler model...")
    # Load the scaler model from the MLflow model registry
    scaler = MLFlowModelLoader("scaler model").load_model(path=_scaler_artifact_path, run_id=mlflow_run_id)
    logger.info("Scaled the features.")
    return _scale_features(features_df, scaler=scaler)


def fit_transform_scaler(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    assert config.de.is_scaler_enabled, "Scaler is not enabled"
    assert_numerical_features(features_df)
    logger.info("Fitting a scaler model...")
    match config.de.scaler_method:
        case "StandardScaler":
            scaler = StandardScaler(**config.de.scaler_params)
        case "MinMaxScaler":
            params = config.de.scaler_params.copy()
            if "feature_range" in params:
                params["feature_range"] = tuple(params["feature_range"])
            scaler = MinMaxScaler(**params)
        case "RobustScaler":
            params = config.de.scaler_params.copy()
            if "quantile_range" in params:
                params["quantile_range"] = tuple(params["quantile_range"])
            scaler = RobustScaler(**params)
        case "MaxAbsScaler":
            scaler = MaxAbsScaler(**config.de.scaler_params)
        case _:
            raise ValueError(f"Unknown scaler method: {config.de.scaler_method}")
    scaled_features = scaler.fit_transform(features_df)
    logger.info("Scaled the features.")
    # Save the scaler model to MLFlow registry
    MLFlowModelLogger(scaler, "scaler model").log_model(_scaler_artifact_path)
    return _scale_features(features_df, scaled_features=scaled_features)
