import importlib
import logging
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.de_config import CustomFeatureConfig
from claim_modelling_kedro.pipelines.p06_data_engineering.utils.utils import assert_features_dtypes
from claim_modelling_kedro.pipelines.utils.mlflow_model import MLFlowModelLoader, MLFlowModelLogger
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path

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
            logger.info(f"Fitting {model.ftr_config.model_class} for a new feature {model.ftr_config.name}...")
            model.fit(features_df)
            logger.info(f"Fitted {model.ftr_config.model_class}.")

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


def create_category_feature(
    features_df: pd.DataFrame,
    column: str,
    intervals: list,
    labels: list = None,
    right: bool = False,
    new_column: str = None
) -> pd.DataFrame:
    # Generowanie etykiet, jeśli nie podano
    if labels is None:
        labels = []
        for i in range(len(intervals) - 1):
            left = intervals[i]
            right_ = intervals[i + 1]
            left_str = "-infty" if left == float("-inf") else str(left)
            right_str = "infty" if right_ == float("inf") else str(right_)
            if right:
                label = f"({left_str},{right_str}]"
            else:
                label = f"[{left_str},{right_str})"
            labels.append(label)
    # Kategoryzacja
    category = pd.cut(features_df[column], bins=intervals, labels=labels, right=right)
    # Zamiana NaN na None
    category = category.where(~category.isna(), None)
    category_df = pd.DataFrame()
    category_df[new_column or f"{column}Category"] = category
    return category_df


class DrivAgeCreatorModel(CustomFeatureCreatorModel):
    def __init__(self, ftr_config: CustomFeatureConfig):
        super().__init__(ftr_config)

    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        age_intervals = [18, 21, 26, 31, 41, 51, 71, float("inf")]
        return create_category_feature(
            features_df,
            column="DrivAge",
            intervals=age_intervals,
            right=False,
            new_column=self.ftr_config.name # we replace the original column with the new one
        )


class VehAgeCreatorModel(CustomFeatureCreatorModel):
    def __init__(self, ftr_config: CustomFeatureConfig):
        super().__init__(ftr_config)

    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        veh_age_intervals = [0, 1, 10, float("inf")]
        return create_category_feature(
            features_df,
            column="VehAge",
            intervals=veh_age_intervals,
            right=False,
            new_column=self.ftr_config.name # we replace the original column with the new one
        )


class VehPowerCreatorModel(CustomFeatureCreatorModel):
    def __init__(self, ftr_config: CustomFeatureConfig):
        super().__init__(ftr_config)

    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    def _transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        veh_power_intervals = list(range(4, 10)) + [float("inf")]
        return create_category_feature(
            features_df,
            column="VehAge",
            intervals=veh_power_intervals,
            right=False,
            new_column=self.ftr_config.name # we replace the original column with the new one
        )
