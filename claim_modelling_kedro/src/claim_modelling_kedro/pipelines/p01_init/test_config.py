from dataclasses import dataclass
from typing import Dict


@dataclass
class TestConfig:
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["test"]
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]
