import json
import logging
import os
import tempfile
from typing import Dict

import mlflow
import numpy as np
import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils.utils import assert_categorical_features, escape_feature_name

logger = logging.getLogger(__name__)

# Define the artifact path for categories_dct
_cat_reducer_categories_dct_artifact_path = "model_de/cat_reducer"
_categories_dct_filename = "categories_dct.json"


def _reduce_categories(features_df: pd.DataFrame, categories_dct: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    logger.info("Reducing categories...")
    cat_reduced_features_df = features_df.copy()
    for col, dct in categories_dct.items():
        cat_reduced_features_df[col] = None
        for old_val, new_val in dct.items():
            cat_reduced_features_df.loc[features_df[col] == old_val, col] = new_val
    logger.info("Reduced the categories.")
    return cat_reduced_features_df


def reduce_categories_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                      mlflow_run_id: str = None) -> pd.DataFrame:
    assert config.de.is_cat_reducer_enabled, "Category reducer is not enabled"
    assert_categorical_features(features_df)
    logger.info("Loading the reducer model...")
    # Download the categories dict from the MLflow server, reduce the categories and return the reduced features
    if mlflow_run_id is None:
        mlflow_run_id = mlflow.active_run().info.run_id
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the artifact from the MLflow server
        local_path = mlflow.artifacts.download_artifacts(
            run_id=mlflow_run_id,
            artifact_path=f"{_cat_reducer_categories_dct_artifact_path}/{_categories_dct_filename}",
            dst_path=os.path.join(temp_dir, _categories_dct_filename))
        # Load the categories_dct from the JSON file
        with open(local_path, "r") as f:
            categories_dct = json.load(f)
            logger.info("Successfully loaded the reducer model.")
            # Reduce the categories and return the reduced features
            return _reduce_categories(features_df, categories_dct)


def fit_transform_categories_reducer(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    assert config.de.is_cat_reducer_enabled, "Category reducer is not enabled"
    assert_categorical_features(features_df)
    logger.info("Fitting the categories reducer...")
    categories_dct = dict()
    for col in features_df.columns:
        # Get the value counts of the column
        counts = features_df[col].value_counts().sort_values(ascending=False)
        # Create an argsort array for the original index of the column (ObsKeys)
        index_order = np.argsort(counts.index)
        # Sort the series by values, but use the index_order for equal values
        counts = counts.iloc[np.lexsort((index_order, -counts.values))]
        logger.debug(f"counts:\n{counts}")
        # Create a dictionary with the mapping of the categories
        map_dct = dict()
        # Remove the least frequent categories that exceed the reducer_max_categories
        if config.de.reducer_max_categories is not None and len(counts) > config.de.reducer_max_categories:
            logger.debug(
                f"config.de.reducer_join_exceeding_max_categories: {config.de.reducer_join_exceeding_max_categories}")
            if config.de.reducer_join_exceeding_max_categories is not None:
                logger.debug(f"counts[config.de.reducer_max_categories:]:\n{counts[config.de.reducer_max_categories:]}")
                map_dct.update({val: escape_feature_name(config.de.reducer_join_exceeding_max_categories) for val in
                                counts[config.de.reducer_max_categories:].index.values})
                logger.debug(f"map_dct:\n{map_dct}")
            counts = counts[:config.de.reducer_max_categories]
            logger.debug(f"counts:\n{counts}")
        # If reducer_min_frequency is not None, update the mapping dictionary for infrequent values
        if config.de.reducer_min_frequency is not None:
            # config.de.reducer_join_exceeding_max_categories is superior to config.de.reducer_join_infrequent, i.e.,
            # if both join_exceeding_max_categories and join_infrequent are provided,
            # the join_exceeding_max_categories is used instead of join_infrequent.
            if config.de.reducer_max_categories is not None and config.de.reducer_join_exceeding_max_categories is not None:
                infrequent_val = config.de.reducer_join_exceeding_max_categories
                logger.warning(config.de.join_exceeding_and_join_infrequent_values_warning)
            else:
                infrequent_val = config.de.reducer_join_infrequent
            if infrequent_val is not None:
                map_dct.update({val: escape_feature_name(infrequent_val) for val in
                                counts[counts.values < config.de.reducer_min_frequency].index.values})
                logger.debug(f"map_dct:\n{map_dct}")
            counts = counts[counts.values >= config.de.reducer_min_frequency]
            logger.debug(f"counts:\n{counts}")
        # Add the remaining categories to the map_dct as those which are not changed
        map_dct.update({val: escape_feature_name(val) for val in counts.index.values})
        if len(map_dct) > 0:
            categories_dct[col] = map_dct
        logger.debug(f"map_dct:\n{map_dct}")
    logger.info("Fitted the reducer.")
    # Save the categories_dct to mlflow
    logger.info("Saving the reducer...")
    with tempfile.TemporaryDirectory() as tempdir:
        # Save the dictionary to a JSON file
        file_path = os.path.join(tempdir, _categories_dct_filename)
        with open(file_path, "w") as f:
            json.dump(categories_dct, f)
        # Log the JSON file as an artifact
        mlflow.log_artifact(file_path, _cat_reducer_categories_dct_artifact_path)
    logger.info("Saved the reducer.")
    return _reduce_categories(features_df, categories_dct)
