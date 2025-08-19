import logging
import os
import tempfile
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import mlflow
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.blacklist import remove_features_from_blacklist, \
    remove_features_from_mlflow_blacklist, save_features_blacklist_to_mlflow, parse_features_blacklist
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.cat_reducer import fit_transform_categories_reducer, \
    reduce_categories_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.custom_features import fit_transform_custom_features_creator, \
    create_custom_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.imputer import fit_transform_categories_imputer, \
    fit_transform_numerical_imputer, \
    impute_categories_by_mlflow_model, impute_numeric_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.onehot_encoder import fit_transform_one_hot_encoder, \
    one_hot_encode_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.polynomial import fit_transform_features_polynomial, \
    transform_features_polynomial_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.scaler import fit_transform_scaler, \
    scale_features_by_mlflow_model
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.pca import transform_features_pca_by_mlflow_model, \
    fit_transform_features_pca
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.utils import split_categories_and_numerics, join_features_dfs
from claim_modelling_kedro.pipelines.utils.datasets import get_partition, get_mlflow_run_id_for_partition

logger = logging.getLogger(__name__)


# Define the artifact path for transformed features
_de_artifact_path = "model_de"
_transformed_numerical_features_filename = "numerical_features.txt"
_transformed_categorical_features_filename = "categorical_features.txt"


def assert_features_dfs_have_same_indexes(features_df, transformed_features_df):
    assert features_df.index.equals(
        transformed_features_df.index), "Indexes of the transformed features do not match the originals."


def fit_transform_features_part(config: Config, features_df: pd.DataFrame, features_blacklist_text: str,
                                reference_categories: Dict[str, str]) -> pd.DataFrame:
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
        categorical_features_df = fit_transform_one_hot_encoder(config, categorical_features_df, reference_categories)
    else:
        # If OHE is not enabled, ensure that categorical features are of type 'category'
        for col in categorical_features_df.columns:
            if categorical_features_df[col].dtype != "category":
                categorical_features_df[col] = categorical_features_df[col].astype("category")

    # Save the transformed features in MLFlow
    for transformed_features_df, features_name, features_filename in zip(
            [categorical_features_df, numerical_features_df],
            ["categorical", "numerical"],
            [_transformed_categorical_features_filename, _transformed_numerical_features_filename]):
        logger.info(f"Saving the transformed {features_name} features in MLFlow...")
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, features_filename)
            with open(file_path, "w") as f:
                content = "\n".join(transformed_features_df.columns.tolist())
                f.write(content)
            mlflow.log_artifact(file_path, _de_artifact_path)
        logger.info(
            f"Successfully saved the {features_name} features to MLFlow path {os.path.join(_de_artifact_path, features_filename)}")

    # Join categorical and numerical features
    transformed_features_df = join_features_dfs([categorical_features_df, numerical_features_df])

    # Remove features from the blacklist and log the blacklist to MLFlow
    features_blacklist = parse_features_blacklist(features_blacklist_text)
    transformed_features_df = remove_features_from_blacklist(transformed_features_df, features_blacklist)
    save_features_blacklist_to_mlflow(features_blacklist_text)

    # If polynomial features are enabled, transform the features using PolynomialFeatures
    if config.de.is_polynomial_features_enabled:
        transformed_features_df = fit_transform_features_polynomial(config, transformed_features_df)

    # If PCA is enabled, transform the features using PCA
    if config.de.is_pca_enabled:
        transformed_features_df = fit_transform_features_pca(config, transformed_features_df)

    # Check that the indexes of the transformed features match the original features
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
    else:
        # If OHE is not enabled, ensure that categorical features are of type 'category'
        for col in categorical_features_df.columns:
            if categorical_features_df[col].dtype != "category":
                categorical_features_df[col] = categorical_features_df[col].astype("category")

    # Join categorical and numerical features
    transformed_features_df = join_features_dfs([categorical_features_df, numerical_features_df])

    # Remove features from the blacklist
    transformed_features_df = remove_features_from_mlflow_blacklist(transformed_features_df, mlflow_run_id)

    # If polynomial features are enabled, transform the features using PolynomialFeatures
    if config.de.is_polynomial_features_enabled:
        transformed_features_df = transform_features_polynomial_by_mlflow_model(transformed_features_df, mlflow_run_id)



    # If PCA is enabled, transform the features using PCA
    if config.de.is_pca_enabled:
        transformed_features_df = transform_features_pca_by_mlflow_model(transformed_features_df, mlflow_run_id)

    # Check that the indexes of the transformed features match the original features
    assert_features_dfs_have_same_indexes(features_df, transformed_features_df)
    return transformed_features_df


def process_partition(config: Config, part: str, features_df: Dict[str, pd.DataFrame],
                      features_blacklist_text: str, reference_categories: Dict[str, str]) -> Tuple[str, pd.DataFrame]:
    features_part_df = get_partition(features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
    logger.info(f"Fitting transformers on partition '{part}' of the sample dataset...")
    with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
        transformed_part_df = fit_transform_features_part(config, features_part_df,
                                                          features_blacklist_text, reference_categories)
    return part, transformed_part_df


def fit_transform_features(config: Config, features_df: Dict[str, pd.DataFrame],
                           features_blacklist_text: str, reference_categories: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    logger.info(f"Fitting features transformers on the sample dataset...")
    transformed_features_df = {}

    for part in features_df.keys():
        part, transformed_part_df = process_partition(config, part, features_df,
                                                      features_blacklist_text, reference_categories)
        transformed_features_df[part] = transformed_part_df
    # parts_cnt = len(features_df)
    # with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
    #     futures = {
    #         executor.submit(process_partition, config, part, features_df, features_blacklist_text, reference_categories): part
    #         for part in features_df.keys()
    #     }
    #     for future in as_completed(futures):
    #         part, transformed_part_df = future.result()
    #         transformed_features_df[part] = transformed_part_df

    return transformed_features_df


def process_transform_partition(config: Config, part: str, features_df: Dict[str, pd.DataFrame],
                                mlflow_run_id: str) -> Tuple[str, pd.DataFrame]:
    features_part_df = get_partition(features_df, part)
    mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part, parent_mlflow_run_id=mlflow_run_id)
    logger.info(f"Transforming features on partition '{part}'...")
    transformed_part_df = transform_features_by_mlflow_model_part(config, features_part_df, mlflow_subrun_id)
    return part, transformed_part_df


def transform_features_by_mlflow_model(config: Config, features_df: Dict[str, pd.DataFrame],
                                       mlflow_run_id: str = None) -> Dict[str, pd.DataFrame]:
    logger.info(f"Transforming features...")
    transformed_features_df = {}

    parts_cnt = len(features_df)
    with ProcessPoolExecutor(max_workers=min(parts_cnt, 10)) as executor:
        futures = {
            executor.submit(process_transform_partition, config, part, features_df, mlflow_run_id): part
            for part in features_df.keys()
        }
        for future in as_completed(futures):
            part, transformed_part_df = future.result()
            transformed_features_df[part] = transformed_part_df

    return transformed_features_df
