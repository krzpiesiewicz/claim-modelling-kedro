import re
import logging
from typing import List, Tuple, Union

import pandas as pd
from unidecode import unidecode

logger = logging.getLogger(__name__)

# Define the data types for categorical features
_categorical_dtypes: List[Union[str, type]] = ["object", "category"]
_numerical_dtypes: List[Union[str, type]] = ["number"]


def select_categorical_features(features_df: pd.DataFrame) -> pd.DataFrame:
    return features_df.select_dtypes(include=_categorical_dtypes)


def select_numerical_features(features_df: pd.DataFrame) -> pd.DataFrame:
    return features_df.select_dtypes(include=_numerical_dtypes)


def assert_features_dtypes(features_df: pd.DataFrame, allowed_dtypes: List[Union[str, type]] = None,
                           err_msg_heading: str = None) -> None:
    if allowed_dtypes is None:
        allowed_dtypes = _categorical_dtypes + _numerical_dtypes
    if err_msg_heading is None:
        err_msg_heading = "The dataframe contains unsupported data types:"
    # Check if the features_df contains any unknown dtype
    unknown_features_df = features_df.select_dtypes(exclude=allowed_dtypes)
    unknown_dtypes = unknown_features_df.dtypes
    if len(unknown_dtypes) > 0:
        err_msg_lines = [err_msg_heading]
        for i, dtype in enumerate(unknown_dtypes):
            cols = sorted(features_df.select_dtypes(include=[dtype]).columns)
            line = f"  ({i + 1}) dtype {dtype} – columns: {cols[0]}"
            if len(cols) > 1:
                line += f" and {len(cols) - 1} more"
            err_msg_lines.append(line)
        raise ValueError("\n".join(err_msg_lines) + ".\nPlease check the data types of the features.")


def assert_categorical_features(features_df: pd.DataFrame) -> None:
    assert_features_dtypes(features_df, _categorical_dtypes,
                           "The dataframe contains non-categorical features:")


def assert_numerical_features(features_df: pd.DataFrame) -> None:
    assert_features_dtypes(features_df, _numerical_dtypes,
                           "The dataframe contains non-numerical features:")


def split_categories_and_numerics(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Check if the features_df contains any unknown dtype
    assert_features_dtypes(features_df, _numerical_dtypes + _categorical_dtypes,
                           "The dataframe contains unknown data types:")
    # Split the features into categorical and numerical
    numerical_features = select_numerical_features(features_df)
    categorical_features = select_categorical_features(features_df)
    return categorical_features, numerical_features


def join_features_dfs(features_dfs_list: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(features_dfs_list, axis=1)


def escape_feature_name(name: str) -> str:
    # Transliterate to ASCII
    name = unidecode(name)
    # Replace '-' and ' ' with '_'
    name = name.replace('-', '_').replace(' ', '_')
    # Remove all non-alphanumeric characters except '_'
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Replace multiple consecutive underscores with a single underscore
    name = re.sub(r'_+', '_', name)
    return name
