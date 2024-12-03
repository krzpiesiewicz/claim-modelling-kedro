import logging
import re
from abc import ABC
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hyperopt import hp
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import ModelEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric, RootMeanSquaredError, MeanPoissonDeviance, \
    MeanGammaDeviance, NormalizedGiniCoefficient, MeanTweedieDeviance


logger = logging.getLogger(__name__)


class StatsmodelsGLM(PredictiveModel):
    """
    It"s important to note that sm.GLM does not have hyperparameters in
    the same sense as machine learning models like Random Forest or
    Gradient Boosting do. Parameters like learning rate, tree depth, etc.,
    can be tuned in those models. The parameters of sm.GLM are more about
    specifying the structure of the model and the data, rather than
    tuning the model"s learning process.
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

    def metric(self) -> Metric:
        match self.config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                return RootMeanSquaredError(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                return MeanGammaDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_POISSON_GLM:
                return MeanPoissonDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                return NormalizedGiniCoefficient(self.config, pred_col=self.pred_col)
            case _:
                raise ValueError(
                    f"""Family for model {self.config.mdl_info.model} not supported in Stastsmodels GLM. Supported are:
            - {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
            - {ModelEnum.STATSMODELS_GAMMA_GLM},
            - {ModelEnum.STATSMODELS_POISSON_GLM},
            - {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.model = None
        self._min_y = None
        force_min_y_pred = self._hparams.get("force_min_y_pred")
        self._force_min_y_pred = force_min_y_pred if force_min_y_pred != "auto" else False
        match self.config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                self.family = sm.families.Gaussian()
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                link = self._hparams.get("link")
                match link or "inverse":
                    case "log":
                        link = sm.families.links.Log()
                    case "inverse":
                        link = sm.families.links.InversePower()
                        if force_min_y_pred == "auto":
                            self._force_min_y_pred = True
                    case _:
                        raise ValueError(f"Link '{link}' not supported for Gamma model.")
                # var = None
                # if "variance" in kwargs:
                #     match kwargs["variance"]:
                #         case "mu":
                #             var = sm.families.varfuncs.mu
                #         case "mu_squared":
                #             var = sm.families.varfuncs.mu_squared
                #         case _:
                #             match = re.match(r"power\((\d+(\.\d+)?)\)", kwargs["variance"])
                #             if match:
                #                 p = float(match.group(1))
                #                 var = sm.families.varfuncs.Power(p)
                self.family = sm.families.Gamma(link=link)
            case ModelEnum.STATSMODELS_POISSON_GLM:
                self.family = sm.families.Poisson()
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                p = self._hparams.get("power")
                if p is None:
                    raise ValueError("Tweedie model requires power parameter.")
                self.family = sm.families.Tweedie(link=sm.families.links.Power(power=1 / (1 - p)), var_power=p)
            case _:
                raise ValueError(
                    f"""Family for model {self.config.mdl_info.model} not supported in Stastsmodels GLM. Supported families are:
            - \"Gaussian\" for {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
            - \"Gamma\" for {ModelEnum.STATSMODELS_GAMMA_GLM},
            - \"Poisson\" for {ModelEnum.STATSMODELS_POISSON_GLM},
            - \"Tweedie\" fpr {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        logger.debug("_fit")
        if type(target_df) is pd.DataFrame:
            y = target_df[self.target_col]
        if self._force_min_y_pred:
            self._min_y = np.min(y)
        if "sample_weight" in kwargs:
            weights = kwargs["sample_weight"]
            kwargs = {key: val for key, val in kwargs.items() if key != "sample_weight"}
            logger.debug(f"Weights - max: {np.max(weights)}, min: {np.min(weights)}, contains NaN: {np.isnan(weights).any()}")
        else:
            weights = None
        # self.model = sm.GLM(y, features_df, family=self.family, var_weights=weights).fit(**kwargs)
        self.model = sm.GLM(y, features_df, family=self.family).fit(**kwargs)
        self._set_features_importances(np.abs(self.model.params))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(features_df)
        logger.debug(f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        logger.debug(f"_force_min_y_pred: {self._force_min_y_pred}, _min_y: {self._min_y}")
        if self._force_min_y_pred:
            y_pred = np.maximum(y_pred, self._min_y)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        return y_pred

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"link": None, "power": None, "force_min_y_pred": "auto"}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def summary(self):
        return self.model.summary()


class SklearnGLM(SklearnModel, ABC):
    def __init__(self, config: Config, model_class, **kwargs):
        super().__init__(config, model_class, **kwargs)

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        super()._fit(features_df, target_df, **kwargs)
        features_importances = pd.Series(np.abs(self.model.coef_))
        if hasattr(self.model, "feature_names_in_"):
            features_importances.index = self.model.feature_names_in_
        self._set_features_importances(features_importances)


class SklearnPoissonGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=PoissonRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", cls._get_solver_options())
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {}
        hparams["alpha"] = float(hopt_hparams["alpha"])
        hparams["fit_intercept"] = bool(hopt_hparams["fit_intercept"])
        hparams["solver"] = cls._get_solver_options()[int(hopt_hparams["solver"])]
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }


class SklearnGammaGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=GammaRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {
            "alpha": float(hopt_hparams["alpha"]),
            "fit_intercept": bool(hopt_hparams["fit_intercept"]),
            "solver": cls._get_solver_options()[int(hopt_hparams["solver"])]
        }
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }


class SklearnTweedieGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=TweedieRegressor, **kwargs)

    def metric(self) -> Metric:
        hparams = self.get_hparams()
        if "power" in hparams:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=hparams["power"])
        return NormalizedGiniCoefficient(self.config)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "power": hp.uniform("power", 1.0, 2.0),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

    @classmethod
    def get_hparams_from_hyperopt_res(cls, hopt_hparams: Dict[str, Any]) -> Dict[str, Any]:
        hparams = {}
        hparams["alpha"] = float(hopt_hparams["alpha"])
        hparams["fit_intercept"] = bool(hopt_hparams["fit_intercept"])
        hparams["power"] = float(hopt_hparams["power"])
        hparams["solver"] = cls._get_solver_options()[int(hopt_hparams["solver"])]
        return hparams

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "power": 1.5,
            "alpha": 1.0,
            "fit_intercept": True,
            "link": "auto",
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "warm_start": False,
            "verbose": 0
        }
