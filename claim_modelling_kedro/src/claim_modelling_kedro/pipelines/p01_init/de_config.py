import logging
from dataclasses import dataclass
from typing import Dict, List


logger = logging.getLogger(__name__)


@dataclass
class CustomFeatureConfig:
    name: str
    type: str
    single_column: bool
    description: str
    model_class: str

    def __init__(self, feature_params: Dict):
        self.name = feature_params["name"]
        self.type = feature_params["type"]
        self.single_column = feature_params["single_column"]
        self.description = feature_params["description"]
        self.model_class = feature_params["model_class"]


@dataclass
class DataEngineeringConfig:
    mlflow_run_id: str
    is_custom_features_creator_enabled: bool
    custom_features: List[CustomFeatureConfig]
    is_cat_reducer_enabled: bool
    reducer_max_categories: int
    reducer_join_exceeding_max_categories: str
    reducer_min_frequency: int
    reducer_join_infrequent: str
    is_cat_imputer_enabled: bool
    cat_imputer_strategy: str
    cat_imputer_fill_value: str
    is_num_imputer_enabled: bool
    num_imputer_strategy: str
    num_imputer_fill_value: float
    is_ohe_enabled: bool
    ohe_drop_reference_cat: bool
    ohe_drop_first: bool
    ohe_drop_binary: bool
    ohe_max_categories: int
    ohe_min_frequency: int
    is_scaler_enabled: bool
    join_exceeding_and_join_infrequent_values_warning: str = None

    def __init__(self, parameters: Dict):
        params = parameters["data_engineering"]
        self.mlflow_run_id = params["mlflow_run_id"]
        self.is_custom_features_creator_enabled = params["custom_features"]["enabled"]
        self.custom_features = [CustomFeatureConfig(feature_params) for feature_params in
                                params["custom_features"]["features"]]
        self.is_cat_reducer_enabled = params["reduce_categories"]["enabled"]
        self.reducer_max_categories = params["reduce_categories"]["max_categories"]
        self.reducer_min_frequency = params["reduce_categories"]["min_frequency"]
        self.reducer_join_exceeding_max_categories = params["reduce_categories"]["join_exceeding_max_categories"]
        self.reducer_join_infrequent = params["reduce_categories"]["join_infrequent"]
        if self.is_cat_reducer_enabled and all(val is not None for val in
                                               [self.reducer_max_categories, self.reducer_min_frequency,
                                                self.reducer_join_infrequent,
                                                self.reducer_join_exceeding_max_categories]):
            self.join_exceeding_and_join_infrequent_values_warning = (
                "Both parameters join_exceeding_max_categories and join_infrequent are provided "
                "and have different values, "
                f"{self.join_exceeding_max_categories} and {self.join_infrequent} respectively.\n"
                f"Thus, {self.join_exceeding_max_categories} will be used for both infrequent and exceeding categories.")
            logger.warning(self.join_exceeding_and_join_infrequent_values_warning)

        self.is_cat_imputer_enabled = params["cat_ftrs_imputer"]["enabled"]
        self.cat_imputer_strategy = params["cat_ftrs_imputer"]["strategy"]
        self.cat_imputer_fill_value = params["cat_ftrs_imputer"]["fill_value"]
        self.is_num_imputer_enabled = params["num_ftrs_imputer"]["enabled"]
        self.num_imputer_strategy = params["num_ftrs_imputer"]["strategy"]
        self.num_imputer_fill_value = params["num_ftrs_imputer"]["fill_value"]
        self.is_ohe_enabled = params["ohe"]["enabled"]
        self.ohe_drop_reference_cat = params["ohe"]["drop_reference_cat"]
        self.ohe_drop_first = params["ohe"]["drop_first"]
        self.ohe_drop_binary = params["ohe"]["drop_binary"]
        self.ohe_max_categories = params["ohe"]["max_categories"]
        self.ohe_min_frequency = params["ohe"]["min_frequency"]
        self.is_scaler_enabled = params["scaler"]["enabled"]
