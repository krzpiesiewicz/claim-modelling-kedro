import logging

import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.utils import assert_categorical_features, \
    assert_numerical_features
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger

logger = logging.getLogger(__name__)

# Define the artifact path for imputers
_num_imputer_artifact_path = "model_de/num_imputer"
_cat_imputer_artifact_path = "model_de/cat_imputer"


def _impute_features(features_df: pd.DataFrame, imputer: SimpleImputer, fit_transform: bool) -> pd.DataFrame:
    if fit_transform:
        imputed_features = imputer.fit_transform(features_df)
    else:
        imputed_features = imputer.transform(features_df)
    return pd.DataFrame(imputed_features, columns=features_df.columns, index=features_df.index)


def _impute_features_by_mlflow_model(config: Config, features_df: pd.DataFrame, feature_type: str,
                                     mlflow_run_id: str = None) -> pd.DataFrame:
    # Ensure the feature type is either 'numerical' or 'categorical'
    assert feature_type in ["numerical", "categorical"], "Feature type must be either 'numerical' or 'categorical'"
    # Check if the imputer is enabled in the configuration based on the feature type
    is_imputer_enabled = config.de.is_cat_imputer_enabled if feature_type == "categorical" else config.de.is_num_imputer_enabled
    assert is_imputer_enabled, f"{feature_type.capitalize()} imputer is not enabled"
    # Check that the features DataFrame only contains the correct type of features
    assert_dtypes = assert_categorical_features if feature_type == "categorical" else assert_numerical_features
    assert_dtypes(features_df)

    logger.info(f"Imputing {feature_type} features by the MLFlow {feature_type} imputer model...")
    # Define the artifact path for the imputer based on the feature type
    artifact_path = _cat_imputer_artifact_path if feature_type == "categorical" else _num_imputer_artifact_path
    # If no MLflow run ID is provided, use the ID of the currently active run
    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id
    else:
        logger.info(f"Using the provided MLFlow run ID: {mlflow_run_id}.")
    # Load the imputer model from the MLflow model registry
    imputer = MLFlowModelLoader(f"{feature_type} imputer model").load_model(path=artifact_path, run_id=mlflow_run_id)
    # Use the loaded model to impute missing values in the features
    logger.info(f"Imputing {feature_type} features by the loaded model...")
    imputed_features = _impute_features(features_df, imputer, fit_transform=False)
    logger.info(f"Imputed the {feature_type} features.")
    return imputed_features


def impute_categories_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                      mlflow_run_id: str = None) -> pd.DataFrame:
    return _impute_features_by_mlflow_model(config, features_df, "categorical", mlflow_run_id)


def impute_numeric_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                   mlflow_run_id: str = None) -> pd.DataFrame:
    return _impute_features_by_mlflow_model(config, features_df, "numerical", mlflow_run_id)


def _fit_transform_imputer(config: Config, features_df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    # Ensure the feature type is either 'numerical' or 'categorical'
    assert feature_type in ["numerical", "categorical"], "Feature type must be either 'numerical' or 'categorical'"
    # Check if the imputer is enabled in the configuration based on the feature type
    is_imputer_enabled = config.de.is_cat_imputer_enabled if feature_type == "categorical" else config.de.is_num_imputer_enabled
    assert is_imputer_enabled, f"{feature_type.capitalize()} imputer is not enabled"
    # Check that the features DataFrame only contains the correct type of features
    assert_dtypes = assert_categorical_features if feature_type == "categorical" else assert_numerical_features
    assert_dtypes(features_df)

    logger.info(f"Fitting a {feature_type} imputer model...")
    # Create a SimpleImputer instance with the specified strategy and fill value
    imputer_strategy = config.de.cat_imputer_strategy if feature_type == "categorical" else config.de.num_imputer_strategy
    imputer_fill_value = config.de.cat_imputer_fill_value if feature_type == "categorical" else config.de.num_imputer_fill_value
    imputer = SimpleImputer(strategy=imputer_strategy, fill_value=imputer_fill_value)
    # Fit the imputer to the data and transform the data
    imputed_features_df = _impute_features(features_df, imputer, fit_transform=True)

    # Define the artifact path for the imputer based on the feature type
    artifact_path = _cat_imputer_artifact_path if feature_type == "categorical" else _num_imputer_artifact_path
    # Save the imputer model
    MLFlowModelLogger(imputer, f"{feature_type} imputer model").log_model(artifact_path)
    return imputed_features_df


def fit_transform_categories_imputer(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    return _fit_transform_imputer(config, features_df, "categorical")


def fit_transform_numerical_imputer(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    return _fit_transform_imputer(config, features_df, "numerical")
