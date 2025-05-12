import logging
from abc import ABC
from typing import Dict, Any, List, Union, Collection

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyglmnet
from hyperopt import hp
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import ModelEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric, RootMeanSquaredError, MeanPoissonDeviance, \
    MeanGammaDeviance, NormalizedConcentrationIndex, MeanTweedieDeviance

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
        PredictiveModel.__init__(self, config, **kwargs)

    def metric(self) -> Metric:
        match self.config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                return RootMeanSquaredError(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                return MeanGammaDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_POISSON_GLM:
                return MeanPoissonDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                return NormalizedConcentrationIndex(self.config, pred_col=self.pred_col)
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
        self._fit_intercept = self._hparams.get("fit_intercept", True)
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
                link = self._hparams.get("link")
                var_power = self._hparams.get("power")
                if var_power is None:
                    raise ValueError("Tweedie model for Power link requires power parameter.")
                match link or "log":
                    case "log":
                        link = sm.families.links.Log()
                    case "power":
                        if var_power == 1:
                            raise ValueError(
                                "power = 1 is not valid for Tweedie with Power link due to division by zero.")
                        link = sm.families.links.Power(power=1 / (1 - var_power))
                self.family = sm.families.Tweedie(link=link, var_power=var_power)
            case _:
                raise ValueError(
                    f"""Family for model {self.config.mdl_info.model} not supported in Stastsmodels GLM. Supported families are:
            - \"Gaussian\" for {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
            - \"Gamma\" for {ModelEnum.STATSMODELS_GAMMA_GLM},
            - \"Poisson\" for {ModelEnum.STATSMODELS_POISSON_GLM},
            - \"Tweedie\" fpr {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        logger.debug("StatsmodelsGLM _fit called")
        logger.debug(f"{self.get_hparams()=}")
        if type(target_df) is pd.DataFrame:
            y = target_df[self.target_col]
        else:
            y = target_df
        if self._force_min_y_pred:
            self._min_y = np.min(y)
        if self._fit_intercept:
            features_df = sm.add_constant(features_df)
        logger.debug(f"{features_df.head()=}")
        logger.debug(f"{y.head()=}")
        if "sample_weight" in kwargs:
            weights = kwargs["sample_weight"]
            kwargs = {key: val for key, val in kwargs.items() if key != "sample_weight"}
            logger.debug(
                f"Weights - max: {np.max(weights)}, min: {np.min(weights)}, contains NaN: {np.isnan(weights).any()}, contains +/–inf: {np.isinf(weights).any() or np.isneginf(weights).any()}")
            self.model = sm.GLM(y, features_df, family=self.family, var_weights=weights).fit(**kwargs)
        else:
            self.model = sm.GLM(y, features_df, family=self.family).fit(**kwargs)
        logger.debug(f"Model fitted - {self.model.summary()}")
        self._set_features_importances(np.abs(self.model.params))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        logger.debug("StatsmodelsGLM _predict called")
        logger.debug(f"{self.get_hparams()=}")
        if self._fit_intercept:
            features_df = sm.add_constant(features_df)
        logger.debug(f"{features_df.head()=}")
        y_pred = self.model.predict(features_df)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
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
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"link": None, "power": None, "force_min_y_pred": "auto", "fit_intercept": True}

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
        SklearnModel.__init__(self, config=config, model_class=model_class, **kwargs)

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        super()._fit(features_df, target_df, **kwargs)
        features_importances = pd.Series(np.abs(self.model.coef_))
        if hasattr(self.model, "feature_names_in_"):
            features_importances.index = self.model.feature_names_in_
        self._set_features_importances(features_importances)


class SklearnPoissonGLM(SklearnGLM):
    def __init__(self, config: Config, **kwargs):
        SklearnGLM.__init__(self, config=config, model_class=PoissonRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanPoissonDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose"
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", cls._get_solver_options())
        }

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
        logger.debug(f"{kwargs=}")
        SklearnGLM.__init__(self, config=config, model_class=GammaRegressor, **kwargs)

    def metric(self) -> Metric:
        return MeanGammaDeviance(self.config, pred_col=self.pred_col)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose"
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

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
        SklearnGLM.__init__(self, config=config, model_class=TweedieRegressor, **kwargs)

    def metric(self) -> Metric:
        hparams = self.get_hparams()
        if "power" in hparams:
            return MeanTweedieDeviance(self.config, pred_col=self.pred_col, power=hparams["power"])
        return NormalizedConcentrationIndex(self.config)

    @classmethod
    def _get_solver_options(cls) -> List[str]:
        return ["lbfgs", "newton-cholesky"]

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return [
            "alpha",
            "fit_intercept",
            "power",
            "solver",
            "max_iter",
            "tol",
            "warm_start",
            "verbose"
        ]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.loguniform("alpha", -5, 10),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "power": hp.uniform("power", 1.0, 2.0),
            "solver": hp.choice("solver", ["lbfgs", "newton-cholesky"])
        }

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


class PyGLMNetGLM(PredictiveModel):
    def __init__(self, config: Config, **kwargs):
        PredictiveModel.__init__(self, config, **kwargs)

    def metric(self) -> Metric:
        match self.config.mdl_info.model:
            case ModelEnum.PYGLMNET_GAMMA_GLM:
                return MeanGammaDeviance(self.config, pred_col=self.pred_col)
            case ModelEnum.PYGLMNET_POISSON_GLM:
                return MeanPoissonDeviance(self.config, pred_col=self.pred_col)
            case _:
                raise ValueError(f"Family for model {self.config.mdl_info.model} not supported in PyGLMNetGLM.")

    def _updated_hparams(self):
        logger.debug(f"_updated_hparams - self._hparams: {self._hparams}")
        self.model = None
        self._min_y = None
        force_min_y_pred = self._hparams.get("force_min_y_pred")
        self._force_min_y_pred = force_min_y_pred if force_min_y_pred != "auto" else False
        self._alpha = self._hparams.get("alpha")
        self._reg_lambda = self._hparams.get("reg_lambda")
        self._fit_intercept = self._hparams.get("fit_intercept")
        self._max_iter = self._hparams.get("max_iter")
        self._tol = self._hparams.get("tol")
        match self.config.mdl_info.model:
            case ModelEnum.PYGLMNET_POISSON_GLM:
                self._distr = "poisson"
            case ModelEnum.PYGLMNET_GAMMA_GLM:
                self._distr = "gamma"
            case _:
                raise ValueError(
                    f"""Distribution for model {self.config.mdl_info.model} not supported in PyGLMNet. Supported distributions are:
            - \"poisson\" for {ModelEnum.PYGLMNET_POISSON_GLM},
            - \"gamma\" for {ModelEnum.PYGLMNET_GAMMA_GLM}.""")

    def _fit(self, features_df: Union[pd.DataFrame, np.ndarray], target_df: Union[pd.DataFrame, np.ndarray], **kwargs):
        logger.debug("PyGLMNetGLM _fit called")
        logger.debug(f"{self.get_hparams()=}")
        y = target_df[self.target_col].values if isinstance(target_df, pd.DataFrame) else target_df
        x = features_df.values if isinstance(features_df, pd.DataFrame) else features_df

        if self._force_min_y_pred:
            self._min_y = np.min(y)

        if "sample_weight" in kwargs:
            weights = kwargs["sample_weight"]
            weights = weights.values if isinstance(weights, pd.DataFrame) else weights

            # Check if weights are integers (can be float type but must have integer values)
            if not np.all(np.isclose(weights, weights.astype(int))):
                raise ValueError("Sample weights must have integer values.")

            logger.debug(
                f"Weights - max: {np.max(weights)}, min: {np.min(weights)}, contains NaN: {np.isnan(weights).any()}, contains +/–inf: {np.isinf(weights).any() or np.isneginf(weights).any()}"
            )

            # Add more observations to the model according to the weights
            x = np.repeat(x, weights.astype(int), axis=0)
            y = np.repeat(y, weights.astype(int), axis=0)

        self.model = pyglmnet.GLM(
            distr=self._distr,
            alpha=self._alpha,
            reg_lambda=self._reg_lambda,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol
        )
        self.model.fit(x, y)
        logger.debug(f"Model fitted - {self.summary()}")
        self._set_features_importances(np.abs(self.model.beta_))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        x = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
        y_pred = self.model.predict(x)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        logger.debug(f"_force_min_y_pred: {self._force_min_y_pred}, _min_y: {self._min_y}")
        if self._force_min_y_pred:
            y_pred = np.maximum(y_pred, self._min_y)
        logger.debug(
            f"Predictions - max: {np.max(y_pred)}, min: {np.min(y_pred)}, contains NaN: {np.isnan(y_pred).any()}")
        return y_pred

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {
            "alpha": hp.uniform("alpha", 0.0, 1.0),
            "reg_lambda": hp.loguniform("reg_lambda", -5, 2),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
            "max_iter": hp.choice("max_iter", [100, 500, 1000]),
            "tol": hp.choice("tol", [1e-4, 1e-5, 1e-6])
        }

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {
            "alpha": 0.5,
            "reg_lambda": 0.1,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-6
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(config=self.config)

    def summary(self):
        return f"PyGLM intercept: {self.model.beta0_}, coefficients: {self.model.beta_}"
