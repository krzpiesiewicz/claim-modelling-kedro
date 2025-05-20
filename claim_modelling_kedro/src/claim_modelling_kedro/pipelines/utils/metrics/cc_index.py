import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum, MetricType
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric


class NormalizedConcentrationIndex(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_normalized_concentration_index, **kwargs)

    @staticmethod
    def _weighted_concentration_index(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
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

        y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)

        # Compute cumulative sums
        cum_true = np.cumsum(y_true * sample_weight)  # cumulative sums of true * weight
        cum_weight = np.cumsum(sample_weight)  # cumulative sums of weights
        total_true = np.sum(y_true * sample_weight)  # sum of all true * weight
        sum_of_weights = cum_weight.iloc[-1]

        return 1 / 2 - np.sum(cum_true) / total_true / sum_of_weights

    @staticmethod
    def _weighted_normalized_concentration_index(y_true: pd.Series, y_pred: pd.Series, sample_weight: pd.Series = None):
        """
        Compute the normalized weighted Concentration Index.

        Args:
            y_true (pd.Series): Series of true values.
            y_pred (pd.Series): Series of predicted values.
            sample_weight (pd.Series): Series of sample weights. If None, all weights are set to 1.

        Returns:
            float: Normalized weighted Concentration Index.
        """
        concentration_index_pred = NormalizedConcentrationIndex._weighted_concentration_index(y_true, y_pred, sample_weight)
        concentration_index_true = NormalizedConcentrationIndex._weighted_concentration_index(y_true, y_true, sample_weight)

        # Avoid division by zero
        if np.isclose(concentration_index_true, 0, atol=1e-5):
            return 0.0

        return concentration_index_pred / concentration_index_true

    def _get_name(self) -> str:
        return "Normalized Concentration Index"

    def _get_short_name(self) -> str:
        return "NCI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_CONCENTRATION_INDEX
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_CONCENTRATION_INDEX
        return MetricEnum.CONCENTRATION_INDEX

    def is_larger_better(self) -> bool:
        return True
