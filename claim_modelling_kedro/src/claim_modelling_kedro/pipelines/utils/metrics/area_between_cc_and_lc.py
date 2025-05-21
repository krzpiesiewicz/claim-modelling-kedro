import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, MetricType
from claim_modelling_kedro.pipelines.utils.concentration_curve import calculate_concentration_curve
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


class AreaBetweenCCAndLC(Metric):
    """
    Class to compute the area between the Concentration Curve (CC) and the Lorenz Curve (LC).
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_area_between_cc_and_lc, **kwargs)

    @staticmethod
    def _weighted_area_between_cc_and_lc(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the weighted Concentration Index.

        Args:
            y_true (pd.Series): Series of true values.
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: Weighted Concentration Index.
        """
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"

        x_cc, y_cc = calculate_concentration_curve(y_true, y_pred, sample_weight)
        x_lc, y_lc = calculate_concentration_curve(y_pred, y_pred, sample_weight)

        aucc = np.trapz(y=y_cc, x=x_cc)
        aucl = np.trapz(y=y_lc, x=x_lc)

        return aucc - aucl

    def _get_name(self) -> str:
        return "Area Between CC and LC"

    def _get_short_name(self) -> str:
        return "ABC"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_ABC
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_ABC
        return MetricEnum.ABC

    def is_larger_better(self) -> bool:
        return False
