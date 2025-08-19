import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import TargetTransformer


class PowerTargetTransformer(TargetTransformer):
    """
    A transformer that applies a power transformation to the target variable.
    The transformation is defined as y' = y^(1/power), where power is a parameter specified in the configuration.
    The inverse transformation is defined as y = (y')^power.
    """
    def __init__(self, config: Config, target_col: str, pred_col: str = None):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col)
        self.power = self.params.get("power")
        if self.power is None:
            raise ValueError("Parameter power must be specified in the configuration of PowerTargetTransformer.")

    def _fit(self, y_true: pd.Series):
        # No fitting parameters needed for log transformation
        pass

    def _transform(self, y_true: pd.Series) -> pd.Series:
        powered_y = y_true.apply(lambda y: np.power(y, 1 / self.power))
        return powered_y

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        return y_pred.apply(lambda x: np.power(x, self.power))
