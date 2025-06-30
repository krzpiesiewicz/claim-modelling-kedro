from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Target(Enum):
    CLAIMS_NUMBER: str = "claims_number"
    FREQUENCY: str = "frequency"
    TOTAL_AMOUNT: str = "total_claims_amount"
    AVG_CLAIM_AMOUNT: str = "avg_claim_amount"
    PURE_PREMIUM: str = "pure_premium"

class TargetWeight(Enum):
    CLAIMS_NUMBER: str = "claims_number_weight"
    EXPOSURE: str = "exposure_weight"


@dataclass
class ExperimentInfo:
    name: str
    author: str
    target: Target
    target_weight: Optional[TargetWeight]
    description: str

    def __init__(self, parameters: Dict):
        params = parameters["experiment"]
        self.name = params["name"]
        self.author = params["author"]
        self.target = Target(params["target"])
        if not params["weighted_target"]:
            self.target_weight = None
        else:
            match self.target:
                case Target.AVG_CLAIM_AMOUNT:
                    self.target_weight = TargetWeight.CLAIMS_NUMBER
                case Target.CLAIMS_NUMBER | Target.FREQUENCY | Target.TOTAL_AMOUNT | Target.PURE_PREMIUM:
                    self.target_weight = TargetWeight.EXPOSURE
                case _:
                    raise ValueError(f"Unsupported target: {params['target']}")
        self.description = params["description"]
