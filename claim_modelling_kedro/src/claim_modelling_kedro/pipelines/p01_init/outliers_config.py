from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class OutliersPolicy(Enum):
    KEEP: str = "keep"
    DROP: str = "drop"
    CLIP: str = "clip"


@dataclass
class OutliersConfig:
    policy: OutliersPolicy
    lower_bound: float
    upper_bound: float

    def __init__(self, outlier_params: Dict[str, Any]):
        self.policy = OutliersPolicy(outlier_params["policy"])
        self.lower_bound = outlier_params["lower_bound"]
        self.upper_bound = outlier_params["upper_bound"]
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound >= self.upper_bound:
                raise ValueError(
                    f"Lower bound {self.lower_bound} must be less than upper bound {self.upper_bound}."
                )
