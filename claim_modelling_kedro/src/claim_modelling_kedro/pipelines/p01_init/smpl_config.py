from dataclasses import dataclass
from enum import Enum
from typing import Dict

from claim_modelling_kedro.pipelines.p01_init.data_config import DataConfig
from claim_modelling_kedro.pipelines.p01_init.outliers_config import OutliersConfig


class SampleValidationSet(Enum):
    NONE: str = "none"
    CALIB_SET: str = "calib_set"
    SPLIT: str = "split_train_val"


@dataclass
class SamplingConfig:
    included_calib_set_in_train_sample: bool
    random_seed: int
    n_obs: int
    target_ratio: float
    allow_lower_ratio: bool
    include_zeros: bool
    outliers: OutliersConfig
    validation_set: SampleValidationSet
    split_val_random_seed: int
    split_val_size: float

    def __init__(self, parameters: Dict, data: DataConfig):
        params = parameters["sampling"]
        self.included_calib_set_in_train_sample = params["included_calib_set_in_train_sample"]
        if self.included_calib_set_in_train_sample and data.shared_train_calib_set:
            raise ValueError("The calibration set cannot be added to the sampling dataset if the shared_train_calib_set is True.\n"
                             "Only one data.shared_train_calib_set and sampling.included_calib_set_in_train_sample can be set to True at the same time.")
        self.random_seed = params["random_seed"]
        self.n_obs = params["n_obs"]
        self.target_ratio = params["target_ratio"]
        self.allow_lower_ratio = params["allow_lower_ratio"]
        self.include_zeros = params["include_zeros"]
        self.outliers = OutliersConfig(params["outliers"])
        self.validation_set = SampleValidationSet(params["validation"]["val_set"])
        self.split_val_random_seed = params["validation"]["split_train_val"]["random_seed"]
        self.split_val_size = params["validation"]["split_train_val"]["val_size"]
