from typing import List, Tuple

import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, MetricType
from claim_modelling_kedro.pipelines.utils.concentration_curve import calculate_concentration_curve_parts
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


def calculate_area_under_cc_parts(curve_parts: List[Tuple[np.ndarray[float], np.ndarray[float]]]) -> float:
    """
    Calculate the area under the Concentration Curve (CC).

    Args:
        curve_parts (List[Tuple[np.ndarray[float], np.ndarray[float]]]): List of tuples containing x and y coordinates of the Concentration Curve's parts.

    Returns:
        float: Area under the Concentration Curve.
    """
    area = 0
    for x_cc, y_cc in curve_parts:
        area = area + np.trapz(y=y_cc, x=x_cc)
    return area


def calculate_area_under_cc(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> float:
    """
    Calculate the area under the Concentration Curve (CC).

    Args:
        y_true (pd.Series): Series of true values.
        y_pred (pd.Series): Series of predicted values.
        sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

    Returns:
        float: Area under the Concentration Curve.
    """
    curve_parts = calculate_concentration_curve_parts(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    return calculate_area_under_cc_parts(curve_parts)


class LorenzGiniIndex(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._gini_index, **kwargs)

    @staticmethod
    def _gini_index(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> float:
        """
        Compute the Gini Index as the area over Lorenz Curve CC(y_pred, y_pred).

        Args:
            y_true (pd.Series): Series of true values (not used here).
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: Gini Index over Lorenz Curve.
        """
        return 1 - 2 * calculate_area_under_cc(y_pred, y_pred, sample_weight)

    def _get_name(self) -> str:
        return "Gini Index (Lorenz Curve)"

    def _get_short_name(self) -> str:
        return "LCGI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_LC_GINI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_LC_GINI
        return MetricEnum.LC_GINI

    def is_larger_better(self) -> bool:
        return True


class ConcentrationCurveGiniIndex(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._cc_gini_index, **kwargs)

    @staticmethod
    def _cc_gini_index(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> float:
        """
        Compute the Gini Index as the area over concentration curve CC(y_true, y_pred).

        Args:
            y_true (pd.Series): Series of true values.
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: Gini Index over the concentration curve.
        """
        return 1 - 2 * calculate_area_under_cc(y_true, y_pred, sample_weight)

    def _get_name(self) -> str:
        return "Gini Index (Concentration Curve)"

    def _get_short_name(self) -> str:
        return "CCGI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_CC_GINI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_CC_GINI
        return MetricEnum.CC_GINI

    def is_larger_better(self) -> bool:
        return True


class NormalizedConcentrationCurveGiniIndex(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._normalized_cc_gini_index, **kwargs)

    @staticmethod
    def _normalized_cc_gini_index(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> float:
        """
        Compute the Normalized Gini Index over the concentration curve CC(y_true, y_pred).

        Args:
            y_true (pd.Series): Series of true values.
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: Normalized Gini Index over the concentration curve.
        """
        ordered_gini = 1 - 2 * calculate_area_under_cc(y_true, y_pred, sample_weight)
        gini_true = 1 - 2 * calculate_area_under_cc(y_true, y_true, sample_weight)

        # Avoid division by zero
        if np.isclose(gini_true, 0, atol=1e-5):
            return 0.0

        return ordered_gini / gini_true

    def _get_name(self) -> str:
        return "Normalized Gini Index (Concentration Curve)"

    def _get_short_name(self) -> str:
        return "NCCGI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_NORMALIZED_CC_GINI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_NORMALIZED_CC_GINI
        return MetricEnum.NORMALIZED_CC_GINI

    def is_larger_better(self) -> bool:
        return True
