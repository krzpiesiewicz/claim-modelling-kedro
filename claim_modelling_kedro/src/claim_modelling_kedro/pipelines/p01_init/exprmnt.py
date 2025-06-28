from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Target(Enum):
    CLAIMS_NUMBER: str = "claims_number"
    FREQUENCY: str = "frequency"
    TOTAL_AMOUNT: str = "total_claims_amount"
    AVG_CLAIM_AMOUNT: str = "avg_claim_amount"
    PURE_PREMIUM: str = "pure_premium"


@dataclass
class ExperimentInfo:
    name: str
    author: str
    target: Target
    weighted_target: bool
    description: str

    def __init__(self, parameters: Dict):
        params = parameters["experiment"]
        self.name = params["name"]
        self.author = params["author"]
        self.target = Target(params["target"])
        self.weighted_target = params["weighted_target"]
        self.description = params["description"]
