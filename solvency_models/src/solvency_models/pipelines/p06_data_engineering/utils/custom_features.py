import importlib
import logging
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p01_init.de_config import CustomFeatureConfig
from solvency_models.pipelines.p06_data_engineering.utils.utils import assert_features_dtypes
from solvency_models.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger
from solvency_models.pipelines.utils.utils import get_class_from_path

logger = logging.getLogger(__name__)

# Define the artifact path for custom features creator
_custom_features_creator_artifact_path = "model_de/custom_features_creator"


class CustomFeatureCreatorModel(ABC):
    def __init__(self, ftr_config: CustomFeatureConfig):
        self.ftr_config = ftr_config

    @abstractmethod
    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        custom_features_df = self._transform(features_df)
        assert_features_dtypes(custom_features_df,
                               err_msg_heading="The custom features contains unsupported dtypes:")
        return custom_features_df

    @abstractmethod
    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        pass


class AllCustomFeaturesCreatorModel():
    models: List[CustomFeatureCreatorModel]
    def __init__(self, config: Config):
        self.models = [get_class_from_path(ftr_conf.model_class)(ftr_conf) for ftr_conf in config.de.custom_features]

    def fit(self, features_df: pd.DataFrame) -> None:
        for model in self.models:
            model.fit(features_df)

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        custom_features_dfs = [model.transform(features_df) for model in self.models]
        return pd.concat([features_df] + custom_features_dfs, axis=1)


def _create_custom_features(features_df: pd.DataFrame, creator) -> pd.DataFrame:
    logger.info("Creating the custom features...")
    custom_features_df = creator.transform(features_df)
    logger.info("Created the custom features.")
    return custom_features_df


def create_custom_features_by_mlflow_model(config: Config, features_df: pd.DataFrame,
                                           mlflow_run_id: str = None) -> pd.DataFrame:
    logger.info("Creating custom features by the MLFlow scaler model...")
    # Load the scaler model from the MLflow model registry
    creator = MLFlowModelLoader("custom features creator").load_model(path=_custom_features_creator_artifact_path,
                                                                      run_id=mlflow_run_id)
    return _create_custom_features(features_df, creator=creator)


def fit_transform_custom_features_creator(config: Config, features_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Fitting a custom feature creator...")
    creator = AllCustomFeaturesCreatorModel(config)
    creator.fit(features_df)
    logger.info("Fitted the creator.")
    # Save the scaler model to MLFlow registry
    MLFlowModelLogger(creator, "custom features creator").log_model(_custom_features_creator_artifact_path)
    return _create_custom_features(features_df, creator=creator)


class DrivAgeCategoryCreatorModel(CustomFeatureCreatorModel):
    def __init__(self, ftr_config: CustomFeatureConfig):
        super().__init__(ftr_config)

    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return self._age_category(features_df)

    def _age_category(self, features_df: pd.DataFrame) -> pd.DataFrame:
        # Define the age intervals
        age_intervals = [18, 21, 26, 31, 41, 51, 71, float("inf")]

        # Generate age_labels based on age_intervals
        age_labels = [f"{age_intervals[i]}-{age_intervals[i + 1] - 1}" for i in range(len(age_intervals) - 1)]
        # Replace '-inf' with '-infty' in the last label
        age_labels[-1] = age_labels[-1].replace("inf", "infty")

        # Create a new DataFrame 'DrivAgeCategory' by cutting the 'DrivAge' column into intervals
        driv_age_category_df = pd.DataFrame()
        driv_age_category_df["DrivAgeCategory"] = pd.cut(features_df["DrivAge"], bins=age_intervals, labels=age_labels,
                                                         right=False)
        return driv_age_category_df
