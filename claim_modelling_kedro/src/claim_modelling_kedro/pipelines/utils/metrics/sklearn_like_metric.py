from abc import ABC
from functools import partial
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd
import sklearn
import wcorr

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import TWEEDIE_DEV, EXP_WEIGHTED_TWEEDIE_DEV, \
    CLNB_WEIGHTED_TWEEDIE_DEV, MetricEnum
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index


class SklearnLikeMetric(Metric, ABC):
    def __init__(self, config: Config, pred_col: str, sklearn_like_metric: Callable = None,
                 exposure_weighted=False, claim_nb_weighted=False):
        super().__init__(config=config, exposure_weighted=exposure_weighted, claim_nb_weighted=claim_nb_weighted)
        self.pred_col = pred_col
        self.target_col = config.mdl_task.target_col
        self.sklearn_like_metric = sklearn_like_metric

    def eval(self, target_df: Union[pd.DataFrame, np.ndarray], prediction_df: Union[pd.DataFrame, np.ndarray],
             sample_weight: Optional[Union[pd.Series, np.ndarray]] = None) -> float:
        assert self.sklearn_like_metric is not None
        y_true = target_df[self.target_col] if isinstance(target_df, pd.DataFrame) else target_df
        y_pred = prediction_df[self.pred_col] if isinstance(prediction_df, pd.DataFrame) else prediction_df
        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                sample_weight = sample_weight.values
        else:
            if isinstance(target_df, pd.DataFrame):
                sample_weight = get_sample_weight(self.config, target_df)
            else:
                sample_weight = np.ones_like(y_true)
        return self.sklearn_like_metric(y_true, y_pred, sample_weight=sample_weight)


class MeanAbsoluteError(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.mae, **kwargs)

    @staticmethod
    def mae(y_true, y_pred, **kwargs):
        mse = sklearn.metrics.mean_absolute_error(y_true, y_pred, **kwargs)
        return np.sqrt(mse)

    def _get_name(self) -> str:
        return "Mean Absolute Error"

    def _get_short_name(self) -> str:
        return "MAE"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_MAE
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_MAE
        return MetricEnum.MAE

    def is_larger_better(self) -> bool:
        return False


class RootMeanSquaredError(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.rmse, **kwargs)

    @staticmethod
    def rmse(y_true, y_pred, **kwargs):
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred, **kwargs)
        return np.sqrt(mse)

    def _get_name(self) -> str:
        return "Root Mean Squared Error"

    def _get_short_name(self) -> str:
        return "RMSE"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_RMSE
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_RMSE
        return MetricEnum.RMSE

    def is_larger_better(self) -> bool:
        return False


class R2(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.r2_score, **kwargs)

    def _get_name(self) -> str:
        return "R2"

    def _get_short_name(self) -> str:
        return "R2"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_R2
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_R2
        return MetricEnum.R2

    def is_larger_better(self) -> bool:
        return True


class MeanBiasDeviation(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.mbd, **kwargs)

    @staticmethod
    def mbd(y_true, y_pred, sample_weight=None, **kwargs):
        """
        Calculate the Mean Bias Deviation (MBD) with optional sample weights.

        Args:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted target values.
            sample_weight (array-like, optional): Weights for each sample. Defaults to None.

        Returns:
            float: Weighted or unweighted Mean Bias Deviation.
        """
        diff = y_true - y_pred
        if sample_weight is None:
            # Unweighted mean
            return np.mean(diff)
        else:
            # Weighted mean
            sample_weight = np.asarray(sample_weight)
            return np.average(diff, weights=sample_weight)

    def _get_name(self) -> str:
        return "Mean Bias Deviation"

    def _get_short_name(self) -> str:
        return "MBD"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_MBD
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_MBD
        return MetricEnum.MBD

    def is_larger_better(self) -> bool:
        raise Exception(
            f"{self.get_name()} is not a maximization nor minimization metric - cannot be used as a loss function.")


class MeanPoissonDeviance(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_poisson_deviance, **kwargs)

    def _get_name(self) -> str:
        return "Mean Poisson Deviance"

    def _get_short_name(self) -> str:
        return "MPD"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_POISSON_DEV
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_POISSON_DEV
        return MetricEnum.POISSON_DEV

    def is_larger_better(self) -> bool:
        return False


class MeanGammaDeviance(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=sklearn.metrics.mean_gamma_deviance, **kwargs)

    def _get_name(self) -> str:
        return "Mean Gamma Deviance"

    def _get_short_name(self) -> str:
        return "MGD"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_GAMMA_DEV
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_GAMMA_DEV
        return MetricEnum.GAMMA_DEV

    def is_larger_better(self) -> bool:
        return False


class MeanTweedieDeviance(SklearnLikeMetric):
    def __init__(self, config: Config, power: float, **kwargs):
        super().__init__(config, **kwargs)
        self.power = power
        self.sklearn_like_metric = partial(sklearn.metrics.mean_tweedie_deviance, power=self.power)

    def _get_name(self) -> str:
        return f"Mean Tweedie Deviance (p={self.power})"

    def _get_short_name(self) -> str:
        return f"MTD_{self.power}"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return EXP_WEIGHTED_TWEEDIE_DEV(self.power)
        if self.claim_nb_weighted:
            return CLNB_WEIGHTED_TWEEDIE_DEV(self.power)
        return TWEEDIE_DEV(self.power)

    def is_larger_better(self) -> bool:
        return False


class SpearmanCorrelation(SklearnLikeMetric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_spearman, **kwargs)

    def _get_name(self) -> str:
        return "Spearman Correlation"

    def _get_short_name(self) -> str:
        return "SC"

    def get_enum(self) -> MetricEnum:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_SPEARMAN
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_SPEARMAN
        return MetricEnum.SPEARMAN

    @staticmethod
    def _weighted_spearman(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(y_true, y_pred, sample_weight)
        y_true = pd.Series(y_true.values)
        y_pred = pd.Series(y_pred.values)
        sample_weight = pd.Series(sample_weight.values)
        return wcorr.WeightedCorr(x=y_true, y=y_pred, w=sample_weight)(method="spearman")

    def is_larger_better(self) -> bool:
        return True


