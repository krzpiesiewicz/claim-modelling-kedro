import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.transform_target import TargetTransformer


class PowerTargetTransformer(TargetTransformer):
    """
    A transformer that applies a logarithmic transformation to the target variable.
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
        sqrt_y = y_true.apply(lambda y: np.power(y, 1 / self.params["power"]))
        return sqrt_y

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        return y_pred.apply(lambda x: np.power(x, self.params["power"]))
