import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p07_data_science.select import SelectorModel


class LassoFeatureSelector(SelectorModel):
    def __init__(self, config: Config, alpha: float = 0.1, fit_intercept: bool = False):
        super().__init__(config=config)
        self.model = Lasso(max_iter=self.max_iter, alpha=alpha, fit_intercept=fit_intercept,
                           copy_X=True)

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        x = features_df.values
        y = target_df[self.target_col].values
        y = y - np.mean(y)
        y = y / np.std(y)
        y += 1
        self.model.fit(x, y)
        feature_importances = pd.Series(np.abs(self.model.coef_),
                                        index=pd.Index(features_df.columns, name="feature"))
        self._set_features_importances(feature_importances)
        self._set_selected_features_from_importances()
