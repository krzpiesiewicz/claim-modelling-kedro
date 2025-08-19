from typing import Dict, Any
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p08_model_calibration.calibration_model import CalibrationModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric, RootMeanSquaredError


class PureCalibration(CalibrationModel):
    def __init__(self, config: Config, **kwargs):
        CalibrationModel.__init__(self, config=config, **kwargs)

    def _fit(self, pure_predictions_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        pass

    def _predict(self, pure_predictions_df: pd.DataFrame) -> pd.Series:
        return pure_predictions_df[self.pure_pred_col]

    def metric(self) -> Metric:
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {}

    def _updated_hparams(self):
        pass

    def summary(self):
        return f"Pure calibration model - returns pure predictions from {self.pure_pred_col} column."

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)
