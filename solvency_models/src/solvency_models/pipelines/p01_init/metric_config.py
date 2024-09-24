import re
from dataclasses import dataclass
from enum import Enum

# Common trait for all metric types
class MetricType:
    pass


class MetricEnum(MetricType, Enum):
    POISSON_DEV: str = "poisson_deviance"
    WEIGHTED_POISSON_DEV: str = "weighted_poisson_deviance"
    GAMMA_DEV: str = "gamma_deviance"
    WEIGHTED_GAMMA_DEV: str = "weighted_gamma_deviance"
    TWEEDIE_DEV: str = "tweedie_deviance"
    WEIGHTED_TWEEDIE_DEV: str = "weighted_tweedie_deviance"
    RMSE: str = "rmse"
    WEIGHTED_RMSE: str = "weighted_rmse"
    R2: str = "r2"
    WEIGHTED_R2: str = "weighted_r2"
    MBD: str = "mbd"
    WEIGHTED_MBD: str = "weighted_mbd"
    SPEARMAN: str = "spearman_correlation"
    SPEARMAN_RANK: str = "spearman_rank_correlation"
    GINI: str = "gini"
    WEIGHTED_GINI: str = "weighted_gini"

@dataclass
class TWEEDIE_DEV(MetricType):
    p: float  # Parameter `p` for the Tweedie distribution

    def serialize(self) -> str:
        """
        Serializes the TweedieDev instance to the format 'TWEEDIE_DEV_p'.
        """
        return f"TWEEDIE_DEV_{self.p}"

    @classmethod
    def deserialize(cls, serialized_str: str):
        """
        Deserializes a string in the format 'TWEEDIE_DEV_p' back into a TweedieDev instance.
        """
        # Use regular expression to extract the p value from the string
        for regex_temp in [r"tweedie_deviance\((\d+(\.\d+)?)\)", r"TWEEDIE_DEV_(\d+(\.\d+)?)"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                p_value = float(match.group(1))  # Convert matched p value to float
                return cls(p=p_value)
        raise ValueError(f"Invalid format for TWEEDIE_DEV: {serialized_str}")


@dataclass
class WEIGHTED_TWEEDIE_DEV(MetricType):
    p: float  # Parameter `p` for weighted Tweedie distribution

    def serialize(self) -> str:
        return f"WEIGHTED_TWEEDIE_DEV_{self.p}"

    @classmethod
    def deserialize(cls, serialized_str: str):
        for regex_temp in [r"weighted_tweedie_deviance\((\d+(\.\d+)?)\)", r"WEIGHTED_TWEEDIE_DEV_(\d+(\.\d+)?)"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                p_value = float(match.group(1))  # Convert matched p value to float
                return cls(p=p_value)
        raise ValueError(f"Invalid format for WEIGHTED_TWEEDIE_DEV: {serialized_str}")
