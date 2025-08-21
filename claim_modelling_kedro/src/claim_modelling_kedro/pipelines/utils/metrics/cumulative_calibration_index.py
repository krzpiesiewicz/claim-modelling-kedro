import numpy as np
import pandas as pd
from typing import Tuple

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricType, MetricEnum
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index
from claim_modelling_kedro.pipelines.utils.metrics.sklearn_like_metric import SklearnLikeMetric


def weighted_cumul_clb_areas(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None) -> Tuple[float, float, float]:
    """
    Compute the areas under:
    - the minimum of line of equity and the cumulative calibration curve
    - the maximum of line of equity and the cumulative calibration curve
    - under the line of equity
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)

    # Weighted cumulative sums
    weighted_true = y_true * sample_weight
    weighted_pred = y_pred * sample_weight

    # Normalize by total
    total_weighted_true = np.sum(weighted_true)
    total_weighted_pred = np.sum(weighted_pred)

    # Compute cumulative proportions
    cum_true = np.cumsum(weighted_true) / total_weighted_true
    cum_pred = np.cumsum(weighted_pred) / total_weighted_pred

    # Include the origin point (0, 0)
    cum_true = np.insert(cum_true, 0, 0)
    cum_pred = np.insert(cum_pred, 0, 0)

    # Compute the area under the minimum of line of equity and the cumulative calibration curve
    area_under_min_equity_and_curve = np.trapz(np.minimum(cum_true, cum_pred), cum_pred)

    # Compute the area under the maximum of line of equity and the cumulative calibration curve
    area_under_max_equity_and_curve = np.trapz(np.maximum(cum_true, cum_pred), cum_pred)

    # Compute the area under the line of equity
    area_under_equity = np.trapz(cum_pred, cum_pred)

    return area_under_min_equity_and_curve, area_under_max_equity_and_curve, area_under_equity


def weighted_cumul_clb_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
    """
    Compute the Cumulative Calibration Index, which measures the inequality of true values distribution
    relative to predictions distributions.
    The Cumulative Calibration Index is defined as an integral of absolute values of the differences beetween cumulative calibration curve and the line of equity

    Args:
        y_true (pd.Series): True values (e.g., claim amounts).
        y_pred (pd.Series): Predicted values (e.g., premium values).
        sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

    Returns:
        float: Cumulative Calibration Index.
    """
    area_under_min_equity_and_curve, area_under_max_equity_and_curve, _ = weighted_cumul_clb_areas(y_true, y_pred,
                                                                                                   sample_weight)
    cci = area_under_max_equity_and_curve - area_under_min_equity_and_curve
    return cci


def weighted_overpricing_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
    """
    Compute the Cumulative Overpricing Index is defined as an integral of the positive differences beetween cumulative calibration curve and the line of equity

    Args:
        y_true (pd.Series): True values (e.g., claim amounts).
        y_pred (pd.Series): Predicted values (e.g., premium values).
        sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

    Returns:
        float: Cumulative Overpricing Index
    """
    area_under_min_equity_and_curve, _, area_under_equity = weighted_cumul_clb_areas(y_true, y_pred, sample_weight)
    coi = area_under_equity - area_under_min_equity_and_curve
    return coi


def weighted_underpricing_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
    """
    Compute the Cumulative Underpricing Index is defined as an integral of the positive differences beetween the line of equity and the cumulative calibration curve

    Args:
        y_true (pd.Series): True values (e.g., claim amounts).
        y_pred (pd.Series): Predicted values (e.g., premium values).
        sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

    Returns:
        float: Cumulative Underpricing Index
    """
    _, area_under_max_equity_and_curve, area_under_equity = weighted_cumul_clb_areas(y_true, y_pred, sample_weight)
    cui = area_under_max_equity_and_curve - area_under_equity
    return cui


class CumulativeCalibrationIndex(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_cumul_clb_idx, **kwargs)

    @staticmethod
    def _weighted_cumul_clb_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the Cumulative Calibration Index, which measures the inequality of true values distribution
        relative to predictions distributions.
        The Cumulative Calibration Index is defined as an integral of absolute values of the differences beetween cumulative calibration curve and the line of equity

        Args:
            y_true (pd.Series): True values (e.g., claim amounts).
            y_pred (pd.Series): Predicted values (e.g., premium values).
            sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

        Returns:
            float: Cumulative Calibration Index.
        """
        return weighted_cumul_clb_idx(y_true, y_pred, sample_weight)

    def _get_name(self) -> str:
        return "Cumulative Calibration Index"

    def _get_short_name(self) -> str:
        return "CCI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_CCI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_CCI
        return MetricEnum.CCI

    def is_larger_better(self) -> bool:
        return False


class CumulativeOverpricingIndex(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_overpricing_idx, **kwargs)

    @staticmethod
    def _weighted_overpricing_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the Cumulative Overpricing Index, which measures the extent of overpricing
        relative to the line of equity.

        Args:
            y_true (pd.Series): True values (e.g., claim amounts).
            y_pred (pd.Series): Predicted values (e.g., premium values).
            sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

        Returns:
            float: Cumulative Overpricing Index.
        """
        return weighted_overpricing_idx(y_true, y_pred, sample_weight)

    def _get_name(self) -> str:
        return "Cumulative Overpricing Index"

    def _get_short_name(self) -> str:
        return "COI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_COI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_COI
        return MetricEnum.COI

    def is_larger_better(self) -> bool:
        return False


class CumulativeUnderpricingIndex(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_underpricing_idx, **kwargs)

    @staticmethod
    def _weighted_underpricing_idx(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the Cumulative Underpricing Index, which measures the extent of underpricing
        relative to the line of equity.

        Args:
            y_true (pd.Series): True values (e.g., claim amounts).
            y_pred (pd.Series): Predicted values (e.g., premium values).
            sample_weight (pd.Series or pd.Series, optional): Weights for observations. Defaults to None.

        Returns:
            float: Cumulative Underpricing Index.
        """
        return weighted_underpricing_idx(y_true, y_pred, sample_weight)

    def _get_name(self) -> str:
        return "Cumulative Underpricing Index"

    def _get_short_name(self) -> str:
        return "CUI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_CUI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_CUI
        return MetricEnum.CUI

    def is_larger_better(self) -> bool:
        return False
