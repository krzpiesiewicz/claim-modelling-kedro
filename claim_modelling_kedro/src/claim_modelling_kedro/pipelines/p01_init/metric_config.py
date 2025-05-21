import re
from dataclasses import dataclass
from enum import Enum


# Common trait for all metric types
class MetricType:
    pass


class MetricEnum(MetricType, Enum):
    POISSON_DEV: str = "poisson_deviance"
    EXP_WEIGHTED_POISSON_DEV: str = "exposure_weighted_poisson_deviance"
    CLNB_WEIGHTED_POISSON_DEV: str = "claim_nb_weighted_poisson_deviance"
    GAMMA_DEV: str = "gamma_deviance"
    EXP_WEIGHTED_GAMMA_DEV: str = "exposure_weighted_gamma_deviance"
    CLNB_WEIGHTED_GAMMA_DEV: str = "claim_nb_weighted_gamma_deviance"
    TWEEDIE_DEV: str = "tweedie_deviance"
    EXP_WEIGHTED_TWEEDIE_DEV: str = "exposure_weighted_tweedie_deviance"
    MAE: str = "mae"
    EXP_WEIGHTED_MAE: str = "exposure_weighted_mae"
    CLNB_WEIGHTED_MAE: str = "claim_nb_weighted_mae"
    RMSE: str = "rmse"
    EXP_WEIGHTED_RMSE: str = "exposure_weighted_rmse"
    CLNB_WEIGHTED_RMSE: str = "claim_nb_weighted_rmse"
    R2: str = "r2"
    EXP_WEIGHTED_R2: str = "exposure_weighted_r2"
    CLNB_WEIGHTED_R2: str = "claim_nb_weighted_r2"
    MBD: str = "mbd"
    EXP_WEIGHTED_MBD: str = "exposure_weighted_mbd"
    CLNB_WEIGHTED_MBD: str = "claim_nb_weighted_mbd"
    SPEARMAN: str = "spearman_correlation"
    EXP_WEIGHTED_SPEARMAN: str = "exposure_weighted_spearman_correlation"
    CLNB_WEIGHTED_SPEARMAN: str = "claim_nb_weighted_spearman_correlation"
    CI: str = "concentration_index"
    EXP_WEIGHTED_CI: str = "exposure_weighted_concentration_index"
    CLNB_WEIGHTED_CI: str = "claim_nb_weighted_concentration_index"
    ABC: str = "area_between_cc_and_lc"
    EXP_WEIGHTED_ABC: str = "exposure_weighted_area_between_cc_and_lc"
    CLNB_WEIGHTED_ABC: str = "claim_nb_weighted_area_between_cc_and_lc"


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

    def __hash__(self):
        return hash(self.serialize())


@dataclass
class EXP_WEIGHTED_TWEEDIE_DEV(MetricType):
    p: float  # Parameter `p` for weighted Tweedie distribution

    def serialize(self) -> str:
        return f"EXP_WEIGHTED_TWEEDIE_DEV_{self.p}"

    @classmethod
    def deserialize(cls, serialized_str: str):
        for regex_temp in [r"exposure_weighted_tweedie_deviance\((\d+(\.\d+)?)\)",
                           r"EXP_WEIGHTED_TWEEDIE_DEV_(\d+(\.\d+)?)"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                p_value = float(match.group(1))  # Convert matched p value to float
                return cls(p=p_value)
        raise ValueError(f"Invalid format for EXP_WEIGHTED_TWEEDIE_DEV: {serialized_str}")

    def __hash__(self):
        return hash(self.serialize())


@dataclass
class CLNB_WEIGHTED_TWEEDIE_DEV(MetricType):
    p: float  # Parameter `p` for weighted Tweedie distribution

    def serialize(self) -> str:
        return f"CLNB_WEIGHTED_TWEEDIE_DEV_{self.p}"

    @classmethod
    def deserialize(cls, serialized_str: str):
        for regex_temp in [r"claim_nb_weighted_tweedie_deviance\((\d+(\.\d+)?)\)",
                           r"CLNB_WEIGHTED_TWEEDIE_DEV_(\d+(\.\d+)?)"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                p_value = float(match.group(1))  # Convert matched p value to float
                return cls(p=p_value)
        raise ValueError(f"Invalid format for CLNB_WEIGHTED_TWEEDIE_DEV: {serialized_str}")

    def __hash__(self):
        return hash(self.serialize())
