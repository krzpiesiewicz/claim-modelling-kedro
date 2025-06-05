from dataclasses import dataclass
from typing import Dict

from claim_modelling_kedro.pipelines.p01_init.outliers_config import OutliersConfig


@dataclass
class SamplingConfig:
    use_calib_data: bool
    random_seed: int
    n_obs: int
    target_ratio: float
    allow_lower_ratio: bool
    include_zeros: bool
    outliers: OutliersConfig
    use_calib_data_for_validation: bool
    split_train_val_enabled: bool
    split_val_random_seed: int
    split_val_size: float

    def __init__(self, parameters: Dict):
        params = parameters["sampling"]
        self.use_calib_data = params["use_calib_data"]
        self.random_seed = params["random_seed"]
        self.n_obs = params["n_obs"]
        self.target_ratio = params["target_ratio"]
        self.allow_lower_ratio = params["allow_lower_ratio"]
        self.include_zeros = params["include_zeros"]
        self.outliers = OutliersConfig(params["outliers"])
        self.use_calib_data_for_validation = params["use_calib_data_for_validation"]
        self.split_train_val_enabled = params["split_train_val"]["enabled"]
        if self.split_train_val_enabled and self.use_calib_data_for_validation:
            raise ValueError("If use_calib_data_for_validation is True, split_train_val should be disabled.")
        self.split_val_random_seed = params["split_train_val"]["random_seed"]
        self.split_val_size = params["split_train_val"]["val_size"]
