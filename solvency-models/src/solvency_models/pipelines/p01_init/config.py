from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Target(Enum):
    FREQUENCY: str = "frequency"
    SEVERITY: str = "severity"


@dataclass
class ModelInfo:
    name: str
    author: str
    type: str
    target: Target
    short_description: str
    full_description: str

    def __init__(self, parameters: Dict):
        params = parameters["model"]
        self.name = params["name"]
        self.author = params["author"]
        self.type = params["type"]
        self.target = Target(params["target"])
        self.short_description = params["short_description"]
        self.full_description = params["full_description"]


@dataclass
class DataConfig:
    claims_freq_target_col: str
    claims_severity_target_col: str
    policy_id_col: str
    split_random_seed: int
    calib_size: float
    backtest_size: float

    def __init__(self, parameters: Dict, mdl_info: ModelInfo):
        params = parameters["data"]
        self.claims_freq_target_col = params["claims_freq_target_col"]
        self.claims_severity_target_col = params["claims_severity_target_col"]
        match mdl_info.target:
            case [Target.FREQUENCY]:
                self.target_col = self.claims_freq_target_col
            case [Target.SEVERITY]:
                self.target_col = self.claims_severity_target_col
        self.policy_id_col = params["policy_id_col"]
        self.split_random_seed = params["split_random_seed"]
        self.calib_size = params["calib_size"]
        self.backtest_size = params["backtest_size"]


@dataclass
class ModelTask:
    target_col: str

    def __init__(self, mdl_info: ModelInfo, data: DataConfig):
        match mdl_info.target:
            case Target.FREQUENCY:
                self.target_col = data.claims_freq_target_col
            case Target.SEVERITY:
                self.target_col = data.claims_severity_target_col


@dataclass
class SamplingConfig:
    random_seed: int
    n_obs: int
    target_ratio: float
    include_zeros: bool
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["sampling"]
        self.random_seed = params["random_seed"]
        self.n_obs = params["n_obs"]
        self.target_ratio = params["target_ratio"]
        self.include_zeros = params["include_zeros"]
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]


@dataclass
class DataEngineeringConfig:
    random_seed: int
    n_features: int

    def __init__(self, parameters: Dict):
        params = parameters["data_engineering"]
        self.random_seed = params["random_seed"]
        self.n_features = params["n_features"]


@dataclass
class DataScienceConfig:
    model_random_seed: int
    split_random_seed: int
    test_size: float

    def __init__(self, parameters: Dict):
        params = parameters["data_science"]
        self.model_random_seed = params["model_random_seed"]
        self.split_random_seed = params["split_random_seed"]
        self.test_size = params["test_size"]

class CalibrationMethod(Enum):
    PURE: str = "pure"
    # ISOTONIC: str = "isotonic"


@dataclass
class CalibrationConfig:
    method: CalibrationMethod
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["calibration"]
        self.method = CalibrationMethod(params["method"])
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]


@dataclass
class BacktestConfig:
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["backtest"]
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]


@dataclass
class Config:
    mdl_info: ModelInfo
    data: DataConfig
    smpl: SamplingConfig
    de: DataEngineeringConfig
    ds: DataScienceConfig
    clb: CalibrationConfig
    bt: BacktestConfig

    def __init__(self, parameters: Dict):
        self.mdl_info = ModelInfo(parameters)
        self.data = DataConfig(parameters, mdl_info=self.mdl_info)
        self.mdl_task = ModelTask(mdl_info=self.mdl_info, data=self.data)
        self.smpl = SamplingConfig(parameters)
        self.de = DataEngineeringConfig(parameters)
        self.ds = DataScienceConfig(parameters)
        self.clb = CalibrationConfig(parameters)
        self.bt = BacktestConfig(parameters)
