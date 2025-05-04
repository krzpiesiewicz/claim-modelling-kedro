from dataclasses import dataclass
from typing import Dict

from claim_modelling_kedro.pipelines.p01_init.outliers_config import OutliersConfig


@dataclass
class TestConfig:
    outliers: OutliersConfig

    def __init__(self, parameters: Dict):
        params = parameters["test"]
        self.outliers = OutliersConfig(params["outliers"])
