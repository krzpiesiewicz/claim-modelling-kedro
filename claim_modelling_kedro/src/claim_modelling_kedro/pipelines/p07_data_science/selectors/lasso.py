from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.p07_data_science.select import SelectorModel


class LassoFeatureSelector(SelectorModel):
    def __init__(self, config: Config, alpha: float = 0.1, fit_intercept: bool = False):
        super().__init__(config=config)
        self.model = Lasso(max_iter=self.max_iter, alpha=alpha, fit_intercept=fit_intercept,
                           copy_X=True)

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, sample_train_keys: Optional[pd.Index] = None,
            sample_val_keys: Optional[pd.Index] = None,**kwargs):
        if sample_train_keys is not None:
            features_df = features_df.loc[sample_train_keys, :]
            target_df = target_df.loc[sample_train_keys, :]
        x = features_df.values
        y = target_df[self.target_col].values
        if self.target_col == self.config.data.claims_avg_amount_target_col:
            y = np.log(y)
        else:
            y = np.log1p(y)
        y = y - np.mean(y)
        y = y / np.std(y)
        y += 1
        sample_weight = get_sample_weight(self.config, target_df)
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight.values
        self.model.fit(x, y, **kwargs)
        features_importance = pd.Series(np.abs(self.model.coef_),
                                        index=pd.Index(features_df.columns, name="feature"))
        self._set_features_importance(features_importance)
        self._set_selected_features_from_importance()
