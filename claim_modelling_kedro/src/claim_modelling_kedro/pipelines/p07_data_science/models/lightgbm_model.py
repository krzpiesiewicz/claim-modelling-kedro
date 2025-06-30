from abc import ABC
from typing import Dict, Any, Collection, Union, Optional, List, Tuple

import numpy as np
import pandas as pd
from hyperopt import hp
from lightgbm import LGBMRegressor

from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.utils.metrics.metric import MeanPoissonDeviance, MeanGammaDeviance, \
    MeanTweedieDeviance
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys


class LightGBMRegressorABC(PredictiveModel):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        PredictiveModel.__init__(self, config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def _updated_hparams(self):
        self._random_state = int(self._hparams.get("random_state", 0))
        self._val_size = float(self._hparams.get("val_size", 0.25))
        # Dynamically calculate early_stopping_rounds based on learning_rate
        learning_rate = float(self._hparams.get("learning_rate", 0.1))
        # The lower the learning_rate, the higher early_stopping_rounds should be
        # early_stopping = int(60 / learning_rate)
        # early_stopping = max(30, min(early_stopping, 200))  # Clamp to [30, 200]
        # self._hparams["early_stopping_rounds"] = early_stopping
        # Ensure num_leaves is not greater than 2 ** max_depth
        if "max_depth" in self._hparams:
            max_depth = int(self._hparams["max_depth"])
            max_leaves = 2 ** max_depth
            if "num_leaves" in self._hparams:
                self._hparams["num_leaves"] = int(min(int(self._hparams["num_leaves"]), max_leaves))
        # Cast remaining hyperparameters to int
        for param in ["early_stopping_rounds", "n_estimators", "max_depth", "num_leaves", "min_child_samples", "max_bin", "max_drop"]:
            if param in self._hparams:
                self._hparams[param] = int(self._hparams[param])
        self.model = None

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "learning_rate",
            "n_estimators",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "feature_fraction",
            "reg_alpha",
            "reg_lambda",
            "random_state",
            "min_split_gain",
            "min_child_weight",
            "max_bin",
            "tol",
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        max_depth = hp.quniform("max_depth", 2, 10, 1)
        num_leaves = hp.quniform("num_leaves", 2, 30, 1)
        boosting_types = ["gbdt"]
        space = dict(
            learning_rate=hp.uniform("learning_rate", 0.001, 0.5),
            # n_estimators=hp.quniform("n_estimators", 50, 1500, 50),
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=hp.quniform("min_child_samples", 15, 100, 1),
            subsample=hp.uniform("subsample", 0.5, 1.0),
            # colsample_bytree=hp.uniform("colsample_bytree", 0.1, 1.0),
            # feature_fraction=hp.uniform("feature_fraction", 0.5, 1.0),
            reg_alpha=hp.loguniform("reg_alpha", np.log(1e-7), np.log(1)),
            reg_lambda=hp.loguniform("reg_lambda", np.log(1e-7), np.log(1)),
            min_split_gain=hp.uniform("min_split_gain", 0, 1.0),
            # min_child_weight=hp.loguniform("min_child_weight", np.log(0.001), np.log(100)),
            max_bin=hp.quniform("max_bin", 8, 256, 4),
            boosting_type=hp.choice("boosting_type", boosting_types),
        )
        return space

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            # Default hyperparameters for LightGBM
            "learning_rate": 0.1,
            "n_estimators": 2000,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "feature_fraction": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 0,
            "val_size": 0.25,
            "force_col_wise": True,
            "early_stopping_rounds": 50,
            "objective": None,  # to be set in child classes
            "eval_metric": None,  # to be set in child classes
        }

    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame,
             sample_train_keys: Optional[pd.Index] = None, sample_val_keys: Optional[pd.Index] = None, **kwargs) -> None:
        # Ensure target_df is DataFrame for stratified split
        index = features_df.index.intersection(target_df.index)
        if sample_train_keys is not None:
            index = index.intersection(sample_train_keys.union(sample_val_keys if sample_val_keys is not None else pd.Index([])))
            sample_train_keys = sample_train_keys.intersection(index)
        if sample_train_keys is None:
            sample_train_keys = index.difference(sample_val_keys or pd.Index([]))
        fit_kwargs = kwargs.copy()
        lgbm_params = self.get_hparams().copy()
        lgbm_params.pop("eval_metric", None)
        lgbm_params.pop("val_size", None)
        # Usuwamy tol i greater_is_better całkowicie, nie przekazujemy ich do modelu
        eval_metric = self._hparams.get("eval_metric", None)
        if eval_metric == "cc_gini":
            from claim_modelling_kedro.pipelines.utils.metrics.gini import ConcentrationCurveGiniIndex
            config = self.config
            pred_col = self.pred_col

            def cc_gini_metric(y_true, y_pred, sample_weight=None):
                metric = ConcentrationCurveGiniIndex(config, pred_col=pred_col)
                assert sample_weight is not None
                value = metric.eval(y_true, y_pred, sample_weight=sample_weight)
                is_larger_better = metric.is_larger_better()
                return "cc_gini", value if is_larger_better else -value, is_larger_better

            fit_kwargs["eval_metric"] = cc_gini_metric
        elif eval_metric is not None:
            fit_kwargs["eval_metric"] = eval_metric
        # Remove greater_is_better from fit_kwargs if present
        # Remove tol from fit_kwargs if present
        if sample_val_keys is None:
            split_kwargs = {}
            sample_weight = kwargs.get("sample_weight", None)
            split_kwargs["sample_weight"] = sample_weight
            sample_train_keys, sample_val_keys = get_stratified_train_test_split_keys(
                target_df.loc[index,:],
                stratify_target_col=self.target_col,
                test_size=self._val_size,
                shuffle=True,
                random_seed=self._random_state,
                verbose=False,
                **split_kwargs
            )
        X_train = features_df.loc[sample_train_keys,:]
        y_train = target_df.loc[sample_train_keys,:]
        X_val = features_df.loc[sample_val_keys,:]
        y_val = target_df.loc[sample_val_keys,:]
        y_train = y_train[self.target_col]
        y_val = y_val[self.target_col]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        if "sample_weight" in fit_kwargs:
            fit_kwargs["eval_sample_weight"] = [fit_kwargs["sample_weight"].loc[sample_val_keys]]
            fit_kwargs["sample_weight"] = fit_kwargs["sample_weight"].loc[sample_train_keys]
        self.model = LGBMRegressor(**lgbm_params)
        self.model.fit(X_train, y_train, **fit_kwargs)
        # Set feature importance after fitting
        importance = self.model.feature_importances_
        if isinstance(features_df, pd.DataFrame):
            importance = pd.Series(importance, index=features_df.columns)
        self._set_features_importance(importance)

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        return self.model.predict(features_df)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def summary(self) -> str:
        return f"model: {self.model} with hyper parameters: {self.get_hparams()}"


class LightGBMPoissonRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def metric(self):
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "eval_metric": "poisson",
            "objective": "poisson",
        })
        return defaults


class LightGBMGammaRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col, **kwargs)

    def metric(self):
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "eval_metric": "gamma",
            "objective": "gamma",
        })
        return defaults


class LightGBMTweedieRegressor(LightGBMRegressorABC):
    def __init__(self, config, target_col: str, pred_col: str, **kwargs):
        super().__init__(config=config, target_col=target_col, pred_col=pred_col,  **kwargs)

    def metric(self):
        power = self._hparams.get("tweedie_variance_power", None)
        if power is not None:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=power)
        from claim_modelling_kedro.pipelines.utils.metrics.gini import ConcentrationCurveGiniIndex
        return ConcentrationCurveGiniIndex(self.config)

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        space = super().get_hparams_space()
        space.update({
            "tweedie_variance_power": hp.uniform("tweedie_variance_power", 1.1, 1.9),
        })
        return space

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        defaults = super().get_default_hparams()
        defaults.update({
            "tweedie_variance_power": 1.5,
            "eval_metric": "tweedie",
            "objective": "tweedie",
        })
        return defaults
