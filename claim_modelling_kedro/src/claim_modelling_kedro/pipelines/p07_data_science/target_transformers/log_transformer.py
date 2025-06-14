import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p07_data_science.transform_target import TargetTransformer


class LogTargetTransformer(TargetTransformer):
    """
    A transformer that applies a logarithmic transformation to the target variable.
    """

    def _fit(self, y_true: pd.Series):
        # No fitting parameters needed for log transformation
        pass

    def _transform(self, y_true: pd.Series) -> pd.Series:
        log_y = y_true.apply(lambda y: np.log(1 + y))
        return log_y

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        return y_pred.apply(lambda x: np.exp(x) - 1)
