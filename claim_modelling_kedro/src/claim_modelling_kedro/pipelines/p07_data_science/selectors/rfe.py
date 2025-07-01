import logging

import pandas as pd
from sklearn.feature_selection import RFE

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.utils.weights import get_sample_weight
from claim_modelling_kedro.pipelines.p07_data_science.select import SelectorModel
from claim_modelling_kedro.pipelines.utils.utils import get_class_from_path

logger = logging.getLogger(__name__)


class RecursiveFeatureEliminationSelector(SelectorModel):
    def __init__(self, config: Config):
        super().__init__(config=config)
        if "estimator_kwargs" in config.ds.fs.params:
            self.estimator_kwargs = config.ds.fs.params["estimator_kwargs"]
        if self.estimator_kwargs is None:
            self.estimator_kwargs = {}
        self.estimator = get_class_from_path(config.ds.model_class)(config=config,
                                                                    target_col=self.config.mdl_task.target_col,
                                                                    pred_col=self.config.mdl_task.prediction_col,
                                                                    **self.estimator_kwargs)
        self.selector = None

    def fit(self, features_df, target_df, **kwargs):
        X = features_df
        y = target_df[self.target_col]
        self.selector = RFE(self.estimator, n_features_to_select=self.max_n_features or len(features_df.columns),
                            importance_getter=(lambda estimator: estimator.get_features_importance()))
        sample_weight = get_sample_weight(self.config, target_df)
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self.selector.fit(X, y, **kwargs)
        self._set_selected_features(self.selector.feature_names_in_[self.selector.support_])
        importance = self.selector.estimator_.get_features_importance().copy()
        importance.index = self.selector.feature_names_in_[importance.index]
        self._set_features_importance(importance)
