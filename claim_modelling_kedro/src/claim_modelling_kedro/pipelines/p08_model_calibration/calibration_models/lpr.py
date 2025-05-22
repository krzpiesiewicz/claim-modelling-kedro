from typing import Dict, Any
import numpy as np
import pandas as pd
from skmisc.loess import loess
import hyperopt as hp

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.model import get_sample_weight
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.utils.dataframes import ordered_by_pred_and_hashed_index
from claim_modelling_kedro.pipelines.utils.metrics import RootMeanSquaredError


class LocalPolynomialRegressionCalibration(CalibrationModel):
    """
    Local Polynomial Regression Calibration Model.

    This model uses local polynomial regression (LOESS) to calibrate predictions.
    For each prediction value, a locally-fitted polynomial is used to model the
    relationship between predictions and true target values, allowing for flexible
    calibration.

    Hyperparameters:
        - span (float): Smoothing factor, representing the fraction of data points
          considered in the local fitting. Must be in the range (0, 1]. Default is 0.5.
        - degree (int): Degree of the locally-fitted polynomial. 1 corresponds to
          locally-linear fitting, and 2 corresponds to locally-quadratic fitting.
          The maximum value is 2. Default is 2.
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        sample_weight = get_sample_weight(self.config, target_df)
        y_true, y_pred, sample_weight = ordered_by_pred_and_hashed_index(
            target_df[self.target_col],
            pure_predictions_df[self.pure_pred_col],
            sample_weight
        )
        self._min_pure_y = y_pred.min()
        self._max_pure_y = y_pred.max()

        self._loess_model = loess(
            x=y_pred,
            y=y_true,
            weights=sample_weight,
            span=self._span,
            degree=self._degree,
            normalize=False,
        )
        self._loess_model.fit()

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        y_pred = pure_predictions_df[self.pure_pred_col].values
        y_pred = np.clip(y_pred, self._min_pure_y, self._max_pure_y)
        pred = self._loess_model.predict(y_pred)
        return pd.Series(pred.values, index=pure_predictions_df.index)

    def metric(self):
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)

    def summary(self):
        return f"LocalPolynomialRegressionCalibration: span={self._span}, degree={self._degree}"

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "span": 0.5,
            "degree": 2,
        }

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "span": hp.uniform("span", 0.1, 1.0),
            "degree": hp.choice("degree", [0, 1, 2]),
        }

    def _updated_hparams(self):
        if self._hparams is not None:
            self._span = float(self._hparams.get("span"))
            self._degree = int(self._hparams.get("degree"))
        self._loess_model = None
        self._min_pure_y = None
        self._max_pure_y = None
