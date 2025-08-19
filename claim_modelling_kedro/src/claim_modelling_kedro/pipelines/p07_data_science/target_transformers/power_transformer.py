import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import TargetTransformer


class PowerTargetTransformer(TargetTransformer):
    """
    A transformer that applies a power transformation to the target variable.
    The transformation is defined as y' = (y+y_shift)^(1/power),
    where y is the original target value, y_shift is a constant shift to avoid negative values
    and power is the exponent used in the transformation.
    The inverse transformation is defined as y = (y')^power - y_shift.
    """
    def __init__(self, config: Config, target_col: str, pred_col: str = None):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col)
        self.y_shift = self.params.get("y_shift")
        if self.y_shift is None:
            raise ValueError("Parameter y_shift must be specified in the configuration of PowerTargetTransformer.")
        self.power = self.params.get("power")
        if self.power is None:
            raise ValueError("Parameter power must be specified in the configuration of PowerTargetTransformer.")

    def _fit(self, y_true: pd.Series):
        # No fitting parameters needed for log transformation
        pass

    def _transform(self, y_true: pd.Series) -> pd.Series:
        powered_y = y_true.apply(lambda y: np.power(y + self.y_shift, 1 / self.power))
        return powered_y

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        return y_pred.apply(lambda x: np.power(x, self.power) - self.y_shift)
