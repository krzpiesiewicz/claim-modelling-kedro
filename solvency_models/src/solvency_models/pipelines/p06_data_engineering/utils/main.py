import logging
from typing import Dict

import mlflow
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils.cat_reducer import fit_transform_categories_reducer, \
    reduce_categories_by_mlflow_model
from solvency_models.pipelines.p06_data_engineering.utils.custom_features import fit_transform_custom_features_creator, \
    create_custom_features_by_mlflow_model
from solvency_models.pipelines.p06_data_engineering.utils.imputer import fit_transform_categories_imputer, \
    fit_transform_numerical_imputer, \
    impute_categories_by_mlflow_model, impute_numeric_by_mlflow_model
from solvency_models.pipelines.p06_data_engineering.utils.onehot_encoder import fit_transform_one_hot_encoder, \
    one_hot_encode_by_mlflow_model
from solvency_models.pipelines.p06_data_engineering.utils.scaler import fit_transform_scaler, \
    scale_features_by_mlflow_model
from solvency_models.pipelines.p06_data_engineering.utils.utils import split_categories_and_numerics, join_features_dfs
from solvency_models.pipelines.utils.utils import get_partition, get_mlflow_run_id_for_partition

logger = logging.getLogger(__name__)


def assert_features_dfs_have_same_indexes(features_df, transformed_features_df):
    assert features_df.index.equals(
        transformed_features_df.index), "Indexes of the transformed features do not match the originals."


def fit_transform_features_part(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"features_df:\n{features_df}")
    # Add custom features
    if config.de.is_custom_features_creator_enabled:
        features_df = fit_transform_custom_features_creator(config, features_df)

    # Split features into categorical and numerical
    categorical_features_df, numerical_features_df = split_categories_and_numerics(features_df)
    logger.info(f"""Initial features:
    – {len(numerical_features_df.columns)} numerical features,
    – {len(categorical_features_df.columns)} categorical features.""")
    logger.debug(f"Numerical features: {numerical_features_df.columns}")
    logger.debug(f"Categorical features: {categorical_features_df.columns}")

    # Apply data engineering steps to numerical features
    # Impute missing numerical values
    if config.de.is_num_imputer_enabled:
        numerical_features_df = fit_transform_numerical_imputer(config, numerical_features_df)
    # Scale numerical features
    if config.de.is_scaler_enabled:
        numerical_features_df = fit_transform_scaler(config, numerical_features_df)

    # Apply data engineering steps to categorical features
    # Reduce number of categories
    if config.de.is_cat_reducer_enabled:
        categorical_features_df = fit_transform_categories_reducer(config, categorical_features_df)
    # Impute missing categorical values
    if config.de.is_cat_imputer_enabled:
        categorical_features_df = fit_transform_categories_imputer(config, categorical_features_df)
    # One-hot encode categorical features
    if config.de.is_ohe_enabled:
        categorical_features_df = fit_transform_one_hot_encoder(config, categorical_features_df)

    # Join categorical and numerical features
    transformed_features_df = join_features_dfs([categorical_features_df, numerical_features_df])
    assert_features_dfs_have_same_indexes(features_df, transformed_features_df)
    return transformed_features_df


def transform_features_by_mlflow_model_part(config: Config, features_df: pd.DataFrame,
                                             mlflow_run_id: str = None) -> pd.DataFrame:
    logger.debug(f"features_df:\n{features_df}")
    # Add custom categories
    if config.de.is_custom_features_creator_enabled:
        features_df = create_custom_features_by_mlflow_model(config, features_df, mlflow_run_id)

    # Split features into categorical and numerical
    categorical_features_df, numerical_features_df = split_categories_and_numerics(features_df)

    # Apply data engineering steps to numerical features
    # Impute missing numerical values
    if config.de.is_num_imputer_enabled:
        numerical_features_df = impute_numeric_by_mlflow_model(config, numerical_features_df, mlflow_run_id)
    # Scale numerical features
    if config.de.is_scaler_enabled:
        numerical_features_df = scale_features_by_mlflow_model(config, numerical_features_df, mlflow_run_id)

    # Apply data engineering steps to categorical features
    # Reduce number of categories
    if config.de.is_cat_reducer_enabled:
        categorical_features_df = reduce_categories_by_mlflow_model(config, categorical_features_df, mlflow_run_id)
    # Impute missing categorical values
    if config.de.is_cat_imputer_enabled:
        categorical_features_df = impute_categories_by_mlflow_model(config, categorical_features_df, mlflow_run_id)
    # One-hot encode categorical features
    if config.de.is_ohe_enabled:
        categorical_features_df = one_hot_encode_by_mlflow_model(config, categorical_features_df, mlflow_run_id)

    # Join categorical and numerical features
    transformed_features_df = join_features_dfs([categorical_features_df, numerical_features_df])
    assert_features_dfs_have_same_indexes(features_df, transformed_features_df)
    return transformed_features_df


def fit_transform_features(config: Config, features_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    logger.info(f"Fitting features transformers on the sample dataset...")
    transformed_features_df = {}
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(f"Fitting transformers on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            transformed_features_df[part] = fit_transform_features_part(config, features_part_df)
    return transformed_features_df


def transform_features_by_mlflow_model(config: Config, features_df: Dict[str, pd.DataFrame],
                                       mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    logger.info(f"Transforming features...")
    transformed_features_df = {}
    for part in features_df.keys():
        features_part_df = get_partition(features_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mflow_run_id=mlflow_run_id)
        logger.info(f"Transforming features on partition '{part}'...")
        transformed_features_df[part] = transform_features_by_mlflow_model_part(config, features_part_df, mlflow_subrun_id)
    return transformed_features_df
