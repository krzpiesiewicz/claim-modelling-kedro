import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, MetricType
from claim_modelling_kedro.pipelines.utils.concentration_curve import calculate_concentration_curve, \
    interpolate_to_points
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


class SupremumDiffBetweenCCAndLC(Metric):
    """
    Class to compute the supremum of the absolute difference between the Concentration Curve (CC) and the Lorenz Curve (LC).
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_sup_abs_diff_cc_and_lc, **kwargs)

    @staticmethod
    def _weighted_sup_abs_diff_cc_and_lc(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the supremum of the absolute difference between the Concentration Curve (CC) and the Lorenz Curve (LC).

        Args:
            y_true (pd.Series): Series of true values.
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: The supremum of the absolute difference between CC and LC.
        """
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"


        x_cc, y_cc = calculate_concentration_curve(y_true, y_pred, sample_weight)
        x_lc, y_lc = calculate_concentration_curve(y_pred, y_pred, sample_weight)

        x_interp = np.linspace(0, 1, num=5000)

        x_cc, y_cc = interpolate_to_points(x_cc, y_cc, x_interp, trunc_to_xs=False)
        x_lc, y_lc = interpolate_to_points(x_lc, y_lc, x_interp, trunc_to_xs=False)

        y_abs_diff = np.abs(y_cc - y_lc)
        return np.max(y_abs_diff)

    def _get_name(self) -> str:
        return "Supremum of Difference Between CC and LC"

    def _get_short_name(self) -> str:
        return "SDCL"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_SUP_CL_DIFF
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_SUP_CL_DIFF
        return MetricEnum.SUP_CL_DIFF

    def is_larger_better(self) -> bool:
        return False
