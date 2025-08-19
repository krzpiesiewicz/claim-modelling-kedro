import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import TargetTransformer


class LogTargetTransformer(TargetTransformer):
    """
    A transformer that applies a logarithmic transformation to the target variable.
    """
    def __init__(self, config: Config, target_col: str, pred_col: str = None):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col)
        self.y_shift = self.params.get("y_shift")
        if self.y_shift is None:
            raise ValueError("Parameter y_shift must be specified in the configuration of LogTargetTransformer.")

    def _fit(self, y_true: pd.Series):
        # No fitting parameters needed for log transformation
        pass

    def _transform(self, y_true: pd.Series) -> pd.Series:
        log_y = y_true.apply(lambda y: np.log(self.y_shift + y))
        return log_y

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        return y_pred.apply(lambda x: np.exp(x) - self.y_shift)
