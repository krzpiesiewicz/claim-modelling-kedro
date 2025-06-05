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
        self.split_val_random_seed = params["split_val_random_seed"]
        self.split_val_size = params["split_val_size"]
