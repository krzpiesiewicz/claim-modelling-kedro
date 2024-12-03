from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
import wcorr

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.metric_config import TWEEDIE_DEV, EXP_WEIGHTED_TWEEDIE_DEV, \
    CLNB_WEIGHTED_TWEEDIE_DEV, MetricEnum, MetricType


class Metric(ABC):
    def __init__(self, config: Config, pred_col: str, sklearn_like_metric: Callable = None,
                 exposure_weighted=False, claim_nb_weighted=False):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.pred_col = pred_col
        self.sklearn_like_metric = sklearn_like_metric
        if exposure_weighted and claim_nb_weighted:
            raise ValueError("Only one of exposure_weighted and claim_nb_weighted can be True.")
        self.exposure_weighted = exposure_weighted
        self.claim_nb_weighted = claim_nb_weighted

    @classmethod
    def from_enum(cls, config: Config, enum: MetricEnum, pred_col: str):
        match enum:
            case MetricEnum.MAE:
                return MeanAbsoluteError(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_MAE:
                return MeanAbsoluteError(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_MAE:
                return MeanAbsoluteError(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.RMSE:
                return RootMeanSquaredError(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_RMSE:
                return RootMeanSquaredError(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_RMSE:
                return RootMeanSquaredError(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.R2:
                return R2(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_R2:
                return R2(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_R2:
                return R2(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.MBD:
                return MeanBiasDeviation(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_MBD:
                return MeanBiasDeviation(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_MBD:
                return MeanBiasDeviation(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.POISSON_DEV:
                return MeanPoissonDeviance(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_POISSON_DEV:
                return MeanPoissonDeviance(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_POISSON_DEV:
                return MeanPoissonDeviance(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.GAMMA_DEV:
                return MeanGammaDeviance(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_GAMMA_DEV:
                return MeanGammaDeviance(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_GAMMA_DEV:
                return MeanGammaDeviance(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.SPEARMAN:
                return SpearmanCorrelation(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_SPEARMAN:
                return SpearmanCorrelation(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_SPEARMAN:
                return SpearmanCorrelation(config, pred_col=pred_col, claim_nb_weighted=True)
            case MetricEnum.GINI:
                return NormalizedGiniCoefficient(config, pred_col=pred_col)
            case MetricEnum.EXP_WEIGHTED_GINI:
                return NormalizedGiniCoefficient(config, pred_col=pred_col, exposure_weighted=True)
            case MetricEnum.CLNB_WEIGHTED_GINI:
                return NormalizedGiniCoefficient(config, pred_col=pred_col, claim_nb_weighted=True)
            # Handle the parametrized cases with TweedieDev(p)
            case TWEEDIE_DEV(p):
                return MeanTweedieDeviance(config, pred_col=pred_col, power=p)
            case EXP_WEIGHTED_TWEEDIE_DEV(p):
                return MeanTweedieDeviance(config, pred_col=pred_col, exposure_weighted=True, power=p)
            case CLNB_WEIGHTED_TWEEDIE_DEV(p):
                return MeanTweedieDeviance(config, pred_col=pred_col, claim_nb_weighted=True, power=p)
            case _:
                raise ValueError(f"The metric enum: {enum} is not supported by the Metric class.")

    def get_name(self) -> str:
        name = self._get_name()
        if self.exposure_weighted:
            name = "Exposure-Weighted " + name
        if self.claim_nb_weighted:
            name = "Claim-Number-Weighted " + name
        return name

    def get_short_name(self) -> str:
        name = self._get_short_name()
        if self.exposure_weighted:
            name = "ew" + name
        if self.claim_nb_weighted:
            name = "nw" + name
        return name

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def _get_short_name(self) -> str:
        pass

    @abstractmethod
    def get_enum(self) -> MetricEnum:
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        pass

    def eval(self, target_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
        assert self.sklearn_like_metric is not None
        y_true = target_df[self.target_col]
        y_pred = prediction_df[self.pred_col]

        sample_weight = pd.Series(np.ones_like(y_true), index=y_true.index)
        if self.exposure_weighted:
            sample_weight = sample_weight * target_df[self.config.data.policy_exposure_col]
        if self.claim_nb_weighted:
            sample_weight = sample_weight * np.fmax(target_df[self.config.data.claims_number_target_col], 1)
        return self.sklearn_like_metric(y_true, y_pred, sample_weight=sample_weight)


class MeanAbsoluteError(Metric):
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


class RootMeanSquaredError(Metric):
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


class R2(Metric):
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


class MeanBiasDeviation(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self.mbd, **kwargs)

    @staticmethod
    def mbd(y_true, y_pred, **kwargs):
        return np.mean(y_true - y_pred)

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


class MeanPoissonDeviance(Metric):
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


class MeanGammaDeviance(Metric):
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


class MeanTweedieDeviance(Metric):
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


class SpearmanCorrelation(Metric):
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
        return wcorr.WeightedCorr(x=y_true, y=y_pred, w=sample_weight)(method="spearman")

    def is_larger_better(self) -> bool:
        return True


class NormalizedGiniCoefficient(Metric):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, sklearn_like_metric=self._weighted_normalized_gini, **kwargs)

    @staticmethod
    def _weighted_gini(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):
        assert len(y_true) == len(y_pred)
        if sample_weight is None:
            sample_weight = np.ones_like(y_true)

        all_data = np.asarray(np.c_[y_true, y_pred, sample_weight, np.arange(len(y_true))], dtype=float)
        # Sort by predicted values, then by index in case of ties
        all_data = all_data[np.lexsort((all_data[:, 3], -1 * all_data[:, 1]))]
        weighted_losses = all_data[:, 0] * all_data[:, 2]
        cumulative_weight = all_data[:, 2].sum()

        weighted_gini_sum = np.cumsum(weighted_losses).sum() / weighted_losses.sum()
        weighted_gini_sum -= (cumulative_weight + 1) / 2.0
        return weighted_gini_sum / cumulative_weight

    @staticmethod
    def _weighted_normalized_gini(y_true, y_pred, sample_weight=None):
        return NormalizedGiniCoefficient._weighted_gini(y_true, y_pred, sample_weight) / \
            NormalizedGiniCoefficient._weighted_gini(y_true, y_true, sample_weight)

    def _get_name(self) -> str:
        return "Normalized Gini Coefficient"

    def _get_short_name(self) -> str:
        return "GINI"

    def get_enum(self) -> MetricType:
        if self.exposure_weighted:
            return MetricEnum.EXP_WEIGHTED_GINI
        if self.claim_nb_weighted:
            return MetricEnum.CLNB_WEIGHTED_GINI
        return MetricEnum.GINI

    def is_larger_better(self) -> bool:
        return True
