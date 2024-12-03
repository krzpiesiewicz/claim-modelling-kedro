import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.utils import assert_pandas_no_lacking_indexes, trunc_target_index, \
    preds_as_dataframe_with_col_name


logger = logging.getLogger(__name__)


class CalibrationModel(ABC):
    def __init__(self, config: Config, fit_kwargs: Dict[str, Any] = None):
        self.config = config
        self.target_col = config.mdl_task.target_col
        self.pure_pred_col = config.clb.pure_prediction_col
        self.calib_pred_col = config.clb.calibrated_prediction_col
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self._hparams = None
        self._features_importances = None

    def fit(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray],
            target_df: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs):
        assert_pandas_no_lacking_indexes(pure_predictions_df, target_df, "pure_predictions_df", "target_df")
        target_df = trunc_target_index(pure_predictions_df, target_df)
        self._fit(pure_predictions_df, target_df, **self._fit_kwargs, **kwargs)

    def predict(self, pure_predictions_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        preds = self._predict(pure_predictions_df)
        preds = preds_as_dataframe_with_col_name(pure_predictions_df, preds, self.calib_pred_col)
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
        super().__init__(config, **kwargs)
        self.y_min = None

    def _set_y_min(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        combined_df = pd.concat([pure_predictions_df[[self.pure_pred_col]], target_df[[self.target_col]]], axis=1)
        sorted_df = combined_df.sort_values(by=self.pure_pred_col, ascending=True)
        first_nonzero_position = sorted_df[sorted_df[self.target_col] > 0].index[0]
        pos = first_nonzero_position
        self.y_min = np.mean(sorted_df[self.target_col].iloc[:pos])

    def get_y_min(self) -> float:
        return self.y_min


class CIRModelCenteredIsotonicRegression(IsotonicLikeCalibrationModel):
    def __init__(self, config: Config, hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.hparams = dict(increasing=True, out_of_bounds="clip")
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = CenteredIsotonicRegression(**self.hparams)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        self._set_y_min(pure_predictions_df, target_df)
        x = pure_predictions_df[self.pure_pred_col]
        y = target_df[self.target_col]
        logger.debug(f"_fit- x: {x}\ny: {y}")
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = pure_predictions_df[self.pure_pred_col]
        logger.debug(f"_predict - x: {x}")
        pred = self.model.transform(x)
        pred[pred < self.y_min] = self.y_min
        return pred


class SklearnIsotonicRegression(IsotonicLikeCalibrationModel):
    def __init__(self, config: Config, hparams: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.hparams = dict(increasing=True, out_of_bounds="clip")
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = IsotonicRegression(**self.hparams)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        self._set_y_min(pure_predictions_df, target_df)
        self.hparams.update(y_min=self.y_min)
        self.model = IsotonicRegression(**self.hparams)
        x = pure_predictions_df[self.pure_pred_col]
        y = target_df[self.target_col]
        self.model.fit(x, y)

    def _predict(self, pure_predictions_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        x = pure_predictions_df[self.pure_pred_col]
        pred = self.model.transform(x)
        pred[pred < self.y_min] = self.y_min
        return pred


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
