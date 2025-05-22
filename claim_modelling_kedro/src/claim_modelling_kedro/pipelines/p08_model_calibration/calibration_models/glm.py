from typing import Dict, Any, Union

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.models import StatsmodelsGLM
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel


logger = logging.getLogger(__name__)


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


class SklearnPoissonGLMCalibration(CalibrationModel):
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


class SklearnGammaGLMCalibration(CalibrationModel):
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


class SklearnTweedieGLMCalibration(CalibrationModel):
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
