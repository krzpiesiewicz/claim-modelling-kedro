import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from claim_modelling_kedro.pipelines.p01_init.exprmnt import TargetWeight


# Common trait for all metric types
class MetricType():
    pass

# Metric type for sklearn metrics (metrics that requires pred_col)
class SklearnMetricType(MetricType):
    pass


class SklearnMetricEnum(SklearnMetricType, Enum):
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
    # Calibration metrics
    ABC: str = "area_between_cc_and_lc"
    EXP_WEIGHTED_ABC: str = "exposure_weighted_area_between_cc_and_lc"
    CLNB_WEIGHTED_ABC: str = "claim_nb_weighted_area_between_cc_and_lc"
    SUP_CL_DIFF: str = "supremum_diff_between_cc_and_lc"
    EXP_WEIGHTED_SUP_CL_DIFF: str = "exposure_weighted_supremum_diff_between_cc_and_lc"
    CLNB_WEIGHTED_SUP_CL_DIFF: str = "claim_nb_weighted_supremum_diff_between_cc_and_lc"
    CCI: str = "cumulative_calibration_index"
    EXP_WEIGHTED_CCI: str = "exposure_weighted_cumulative_calibration_index"
    CLNB_WEIGHTED_CCI: str = "claim_nb_weighted_cumulative_calibration_index"
    COI: str = "cumulative_overpricing_index"
    EXP_WEIGHTED_COI: str = "exposure_weighted_cumulative_overpricing_index"
    CLNB_WEIGHTED_COI: str = "claim_nb_weighted_cumulative_overpricing_index"
    CUI: str = "cumulative_underpricing_index"
    EXP_WEIGHTED_CUI: str = "exposure_weighted_cumulative_underpricing_index"
    CLNB_WEIGHTED_CUI: str = "claim_nb_weighted_cumulative_underpricing_index"
    LC_GINI: str = "lc_gini"
    EXP_WEIGHTED_LC_GINI: str = "exposure_weighted_lc_gini"
    CLNB_WEIGHTED_LC_GINI: str = "claim_nb_weighted_lc_gini"
    CC_GINI: str = "cc_gini"
    EXP_WEIGHTED_CC_GINI: str = "exposure_weighted_cc_gini"
    CLNB_WEIGHTED_CC_GINI: str = "claim_nb_weighted_cc_gini"
    NORMALIZED_CC_GINI: str = "normalized_cc_gini"
    EXP_WEIGHTED_NORMALIZED_CC_GINI: str = "exposure_weighted_normalized_cc_gini"
    CLNB_WEIGHTED_NORMALIZED_CC_GINI: str = "claim_nb_weighted_normalized_cc_gini"


@dataclass
class TWEEDIE_DEV(SklearnMetricType):
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
class EXP_WEIGHTED_TWEEDIE_DEV(SklearnMetricType):
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
class CLNB_WEIGHTED_TWEEDIE_DEV(SklearnMetricType):
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


@dataclass
class BinsMetricType(MetricType, ABC):
    n_bins: int  # Parameter `n_bins` for the number of groups that the predictions will be divided into

    @abstractmethod
    def serialize(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized_str: str):
        ...


@dataclass
class MAX_ABS_BIAS_BINS(BinsMetricType):

    def serialize(self) -> str:
        """
        Serializes the MAX_ABS_BIAS_BINS instance to the format 'MAX_ABS_BIAS_{n_bins}_BINS'.
        """
        return f"MAX_ABS_BIAS_{self.n_bins}_BINS"

    @classmethod
    def deserialize(cls, serialized_str: str):
        """
        Deserializes a string in the format 'MAX_ABS_BIAS_{n_bins}_BINS' back into a MAX_ABS_BIAS_BINS instance.
        """
        # Use regular expression to extract the n_bins value from the string
        for regex_temp in [r"max_abs_bias_\((\d+(\.\d+)?)\)_bins", r"MAX_ABS_BIAS_(\d+(\.\d+)?)_BINS"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                n_bins = int(match.group(1))  # Convert matched n_bins value to int
                return cls(n_bins=n_bins)
        raise ValueError(f"Invalid format for MAX_ABS_BIAS_BINS: {serialized_str}")

    def __hash__(self):
        return hash(self.serialize())


@dataclass
class MAX_OVERPRICING_BINS(BinsMetricType):

    def serialize(self) -> str:
        return f"MAX_OVERPRICING_{self.n_bins}_BINS"

    @classmethod
    def deserialize(cls, serialized_str: str):
        for regex_temp in [r"max_overpricing_\((\d+(\.\d+)?)\)_bins", r"MAX_OVERPRICING_(\d+(\.\d+)?)_BINS"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                n_bins = int(match.group(1))  # Convert matched n_bins value to int
                return cls(n_bins=n_bins)
        raise ValueError(f"Invalid format for MAX_OVERPRICING_BINS: {serialized_str}")

    def __hash__(self):
        return hash(self.serialize())


@dataclass
class MAX_UNDERPRICING_BINS(BinsMetricType):

    def serialize(self) -> str:
        return f"MAX_UNDERPRICING_{self.n_bins}_BINS"

    @classmethod
    def deserialize(cls, serialized_str: str):
        for regex_temp in [r"max_underpricing_\((\d+(\.\d+)?)\)_bins", r"MAX_UNDERPRICING_(\d+(\.\d+)?)_BINS"]:
            match = re.match(regex_temp, serialized_str)
            if match:
                n_bins = int(match.group(1))  # Convert matched n_bins value to int
                return cls(n_bins=n_bins)
        raise ValueError(f"Invalid format for MAX_UNDERPRICING_BINS: {serialized_str}")

    def __hash__(self):
        return hash(self.serialize())


def get_weighted_metric_enum(enum: MetricType, weight: TargetWeight) -> MetricType:
    """
    Returns the corresponding weighted metric (exposure or claims number) for the given base metric.
    If the metric does not support weighting or if no weight is specified, the original metric is returned.
    Args:
        enum (MetricType): The base metric to be weighted.
        weight (TargetWeight): The type of weight to apply (exposure or claims number).
    """
    if issubclass(type(enum), BinsMetricType) or weight is None:
        return enum
    if weight == TargetWeight.EXPOSURE:
        if isinstance(enum, TWEEDIE_DEV):
            return EXP_WEIGHTED_TWEEDIE_DEV(enum.p)
        exposure_map = {
            SklearnMetricEnum.POISSON_DEV: SklearnMetricEnum.EXP_WEIGHTED_POISSON_DEV,
            SklearnMetricEnum.GAMMA_DEV: SklearnMetricEnum.EXP_WEIGHTED_GAMMA_DEV,
            SklearnMetricEnum.MAE: SklearnMetricEnum.EXP_WEIGHTED_MAE,
            SklearnMetricEnum.RMSE: SklearnMetricEnum.EXP_WEIGHTED_RMSE,
            SklearnMetricEnum.R2: SklearnMetricEnum.EXP_WEIGHTED_R2,
            SklearnMetricEnum.MBD: SklearnMetricEnum.EXP_WEIGHTED_MBD,
            SklearnMetricEnum.SPEARMAN: SklearnMetricEnum.EXP_WEIGHTED_SPEARMAN,
            SklearnMetricEnum.ABC: SklearnMetricEnum.EXP_WEIGHTED_ABC,
            SklearnMetricEnum.CCI: SklearnMetricEnum.EXP_WEIGHTED_CCI,
            SklearnMetricEnum.COI: SklearnMetricEnum.EXP_WEIGHTED_COI,
            SklearnMetricEnum.CUI: SklearnMetricEnum.EXP_WEIGHTED_CUI,
            SklearnMetricEnum.LC_GINI: SklearnMetricEnum.EXP_WEIGHTED_LC_GINI,
            SklearnMetricEnum.CC_GINI: SklearnMetricEnum.EXP_WEIGHTED_CC_GINI,
            SklearnMetricEnum.NORMALIZED_CC_GINI: SklearnMetricEnum.EXP_WEIGHTED_NORMALIZED_CC_GINI,
        }
        if enum in exposure_map:
            return exposure_map[enum]
        raise ValueError(f"The metric enum: {enum} is not supported for exposure weighting.")
    elif weight == TargetWeight.CLAIMS_NUMBER:
        if isinstance(enum, TWEEDIE_DEV):
            return CLNB_WEIGHTED_TWEEDIE_DEV(enum.p)
        claims_map = {
            SklearnMetricEnum.POISSON_DEV: SklearnMetricEnum.CLNB_WEIGHTED_POISSON_DEV,
            SklearnMetricEnum.GAMMA_DEV: SklearnMetricEnum.CLNB_WEIGHTED_GAMMA_DEV,
            SklearnMetricEnum.MAE: SklearnMetricEnum.CLNB_WEIGHTED_MAE,
            SklearnMetricEnum.RMSE: SklearnMetricEnum.CLNB_WEIGHTED_RMSE,
            SklearnMetricEnum.R2: SklearnMetricEnum.CLNB_WEIGHTED_R2,
            SklearnMetricEnum.MBD: SklearnMetricEnum.CLNB_WEIGHTED_MBD,
            SklearnMetricEnum.SPEARMAN: SklearnMetricEnum.CLNB_WEIGHTED_SPEARMAN,
            SklearnMetricEnum.ABC: SklearnMetricEnum.CLNB_WEIGHTED_ABC,
            SklearnMetricEnum.CCI: SklearnMetricEnum.CLNB_WEIGHTED_CCI,
            SklearnMetricEnum.COI: SklearnMetricEnum.CLNB_WEIGHTED_COI,
            SklearnMetricEnum.CUI: SklearnMetricEnum.CLNB_WEIGHTED_CUI,
            SklearnMetricEnum.LC_GINI: SklearnMetricEnum.CLNB_WEIGHTED_LC_GINI,
            SklearnMetricEnum.CC_GINI: SklearnMetricEnum.CLNB_WEIGHTED_CC_GINI,
            SklearnMetricEnum.NORMALIZED_CC_GINI: SklearnMetricEnum.CLNB_WEIGHTED_NORMALIZED_CC_GINI,
        }
        if enum in claims_map:
            return claims_map[enum]
        raise ValueError(f"The metric enum: {enum} is not supported for claims number weighting.")
    else:
        raise ValueError(f"Unsupported target weight: {weight}. Supported weights are: {TargetWeight.EXPOSURE}, {TargetWeight.CLAIMS_NUMBER}.")
