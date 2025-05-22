import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from hyperopt import hp

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel, get_sample_weight
from claim_modelling_kedro.pipelines.p07_data_science.models import StatsmodelsGLM
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics.metric import Metric, RootMeanSquaredError
from claim_modelling_kedro.pipelines.utils.dataframes import assert_pandas_no_lacking_indexes, trunc_target_index, \
    ordered_by_pred_and_hashed_index

logger = logging.getLogger(__name__)


class CalibrationModel(PredictiveModel, ABC):
    def __init__(self, config: Config, **kwargs):
        logger.debug(f"CalibrationModel.__init__")
        PredictiveModel.__init__(self, config, target_col=config.mdl_task.target_col,
                         pred_col=config.clb.calibrated_prediction_col, **kwargs)
        self.pure_pred_col = config.clb.pure_prediction_col
        self.calib_pred_col = config.clb.calibrated_prediction_col

    def fit(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        logger.debug("CalibrationModel fit called")
        assert_pandas_no_lacking_indexes(pure_predictions_df, target_df, "pure_predictions_df", "target_df")
        target_df = trunc_target_index(pure_predictions_df, target_df)
        super().fit(pure_predictions_df, target_df, **kwargs)

    def predict(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        logger.debug("CalibrationModel predict called")
        preds = super().predict(pure_predictions_df)
        return preds

    def transform(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return self.predict(pure_predictions_df)

    @abstractmethod
    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass


class IsotonicLikeCalibrationModel(CalibrationModel, ABC):
    def __init__(self, config: Config, **kwargs):
        logger.debug(f"IsotonicLikeCalibrationModel.__init__")
        CalibrationModel.__init__(self, config, **kwargs)
        self._y_min = None

    def _set_y_min(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        combined_df = pd.concat([pure_predictions_df[[self.pure_pred_col]], target_df[[self.target_col]]], axis=1)
        sorted_df = combined_df.sort_values(by=self.pure_pred_col, ascending=True)
        logger.debug(f"{sorted_df=}")
        # Find the first non-zero position and replace the values before it (including that position)
        # with the mean of the target values up to that position so that the balance is preserved
        first_nonzero_position = sorted_df[sorted_df[self.target_col] > 0].index[0]
        logger.debug(f"{first_nonzero_position=}")
        pos = first_nonzero_position
        smallest_df = sorted_df.loc[:pos, self.target_col]
        logger.debug(f"{smallest_df=}")
        self._y_min = np.mean(smallest_df)
        logger.debug(f"{self._y_min=}")
        smallest_idx = smallest_df.index
        logger.debug(f"{smallest_idx=}")
        adjusted_target_df = target_df.copy()
        adjusted_target_df.loc[smallest_idx, self.target_col] = self._y_min
        return adjusted_target_df

    def get_y_min(self) -> float:
        return self._y_min

    def metric(self) -> Metric:
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)


class SklearnLikeIsotonicRegression(IsotonicLikeCalibrationModel, SklearnModel):
    def __init__(self, config: Config, model_class, **kwargs):
        IsotonicLikeCalibrationModel.__init__(self, config, call_updated_hparams=False, **kwargs)
        SklearnModel.__init__(self, config=config, model_class=model_class,
                              pred_col=self.pred_col, target_col=self.target_col, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        logger.debug(f"CSklearnLikeIsotonicRegression._fit called with kwargs: {kwargs}")
        if self.force_positive:
            target_df = self._set_y_min(pure_predictions_df, target_df)
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        SklearnModel._fit(self, pure_predictions_df, target_df, **kwargs)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        pred = SklearnModel._predict(self, pure_predictions_df)
        return pred

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.force_positive = self.get_hparams().get("force_positive")
        self._y_min = None
        SklearnModel._updated_hparams(self)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "increasing": True,
            "out_of_bounds": "clip",
            "force_positive": True
        }

    @classmethod
    def _sklearn_hparams_names(cls) -> List[str]:
        return ["increasing", "out_of_bounds"]


class CIRModelCenteredIsotonicRegression(SklearnLikeIsotonicRegression):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=CenteredIsotonicRegression, **kwargs)


class SklearnIsotonicRegression(SklearnLikeIsotonicRegression):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=IsotonicRegression, **kwargs)


class StatsmodelsGLMCalibration(CalibrationModel, StatsmodelsGLM):
    """
    A CalibrationModel that uses StatsmodelsGLM for predictive modeling with calibration capabilities.
    """
    def __init__(self, config: Config, **kwargs):
        # Initialize both parent classes
        CalibrationModel.__init__(self, config, **kwargs)
        StatsmodelsGLM.__init__(self, config, pred_col=self.pred_col, target_col=self.target_col, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        """
        Use the _fit method from StatsmodelsGLM for fitting the calibration model.
        """
        logger.debug("StatsmodelsGLMCalibration _fit called")
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        # Call StatsmodelsGLM's _fit method
        StatsmodelsGLM._fit(self, pure_predictions_df, target_df, **kwargs)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        """
        Use the _predict method from StatsmodelsGLM for making predictions.
        """
        logger.debug("StatsmodelsGLMCalibration _predict called")
        pure_predictions_df = pure_predictions_df[self.pure_pred_col]
        # Call StatsmodelsGLM's _predict method
        return StatsmodelsGLM._predict(self, pure_predictions_df)


class SklearnPoissonGLM(CalibrationModel):
    def __init__(self, config: Config, hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.hparams = dict(fit_intercept=True, alpha=0)
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = PoissonRegressor(**self.hparams)

    def create_features(self, pure_predictions_df: pd.DataFrame) -> pd.DataFrame:
        features_df = pure_predictions_df.loc[:, [self.pure_pred_col]].copy()
        # features_df["log(1+x)"] = np.log1p(features_df[self.pure_pred_col])
        return features_df

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = self.create_features(pure_predictions_df)
        y = target_df[self.target_col]
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = self.create_features(pure_predictions_df)
        return self.model.predict(x)


class SklearnGammaGLM(CalibrationModel):
    def __init__(self, config: Config, hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.hparams = dict(fit_intercept=True, alpha=0)
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = GammaRegressor(**self.hparams)

    def create_features(self, pure_predictions_df: pd.DataFrame) -> pd.DataFrame:
        features_df = pure_predictions_df.loc[:, [self.pure_pred_col]].copy()
        # features_df["log(1+x)"] = np.log1p(features_df[self.pure_pred_col])
        return features_df

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = self.create_features(pure_predictions_df)
        y = target_df[self.target_col]
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = self.create_features(pure_predictions_df)
        return self.model.predict(x)


class SklearnTweedieGLM(CalibrationModel):
    def __init__(self, config: Config, hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.hparams = dict(fit_intercept=True, alpha=0)
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = None
        self.power = None

    def create_features(self, pure_predictions_df: pd.DataFrame) -> pd.DataFrame:
        features_df = pure_predictions_df.loc[:, [self.pure_pred_col]].copy()
        # features_df["log(1+x)"] = np.log1p(features_df[self.pure_pred_col])
        return features_df

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = self.create_features(pure_predictions_df)
        y = target_df[self.target_col]
        self.power = self._hyperopt_tweedie_power(pure_predictions_df, target_df)
        self.hparams.update(power=self.power)
        self.model = TweedieRegressor(**self.hparams)
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = self.create_features(pure_predictions_df)
        return self.model.predict(x)

    def _hyperopt_tweedie_power(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame) -> float:
        """ Finds the best power parameter in (1,2) with hyperopt regarding the GINI metric """
        ...


class EqualBinsMeansCalibration(CalibrationModel):
    def __init__(self, config: Config, n_bins: int = 10, **kwargs):
        super().__init__(config, **kwargs)
        self._n_bins = n_bins
        self._bin_bounds = None
        self._bin_means = None

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        logger.debug("EqualBinsMeansCalibration _fit called")
        sample_weight = get_sample_weight(config=self.config, target_df=target_df)
        y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(
            target_df[self.target_col],
            pure_predictions_df[self.pure_pred_col],
            sample_weight
        )

        # Use qcut with duplicates='drop' to handle repeated prediction values
        try:
            _, self._bin_bounds = pd.qcut(
                y_pred,
                q=self._n_bins,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            self._n_bins = len(self._bin_bounds) - 1
        except ValueError as e:
            logger.warning(f"pd.qcut failed: {e}. Falling back to uniform binning.")
            self._bin_bounds = np.linspace(np.min(y_pred), np.max(y_pred), self._n_bins + 1)
        bin_indices = np.digitize(y_pred, self._bin_bounds[1:-1], right=False)

        # Compute means per bin
        self._bin_means = {}
        for bin_idx in range(self._n_bins):
            bin_mask = bin_indices == bin_idx
            if bin_mask.any():
                self._bin_means[bin_idx] = np.average(y_true[bin_mask], weights=sample_weight[bin_mask])

        # Global mean fallback
        self._global_mean = np.average(y_true, weights=sample_weight)

        logger.debug(f"Fitted {self._n_bins} bins: bounds={self._bin_bounds}, means={self._bin_means}")

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        logger.debug("EqualBinsMeansCalibration _predict called")
        y_pure_pred = pure_predictions_df[self.pure_pred_col].values
        bin_indices = np.digitize(y_pure_pred, self._bin_bounds[1:-1], right=False)
        calibrated = [self._bin_means.get(idx, self._global_mean) for idx in bin_indices]
        return pd.Series(calibrated, index=pure_predictions_df.index)

    def metric(self):
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"n_bins": 10}

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {"n_bins": hp.quniform("n_bins", 2, 100, 1)}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)

    def summary(self):
        return f"EqualBinsMeansCalibration with {self._n_bins} bins.\nBounds: {self._bin_bounds}\nMeans: {self._bin_means}"

    def _updated_hparams(self):
        if self._hparams is not None:
            self._n_bins = int(self._hparams.get("n_bins"))
