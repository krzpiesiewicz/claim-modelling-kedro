from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hyperopt import hp
from sklearn.linear_model import PoissonRegressor

from solvency_models.pipelines.p01_init.config import Config, ModelEnum
from solvency_models.pipelines.p07_data_science.model import PredictiveModel
from solvency_models.pipelines.p07_data_science.models.sklearn_model import SklearnModel


class StatsmodelsGLM(PredictiveModel):
    """
    It's important to note that sm.GLM does not have hyperparameters in
    the same sense as machine learning models like Random Forest or
    Gradient Boosting do. Parameters like learning rate, tree depth, etc.,
    can be tuned in those models. The parameters of sm.GLM are more about
    specifying the structure of the model and the data, rather than
    tuning the model's learning process.
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        match config.mdl_info.model:
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                self.family = sm.families.Gaussian()
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                self.family = sm.families.Gamma()
            case ModelEnum.STATSMODELS_POISSON_GLM:
                self.family = sm.families.Poisson()
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                self.family = sm.families.Tweedie()
            case _:
                raise ValueError(
                    f"""Family for model {config.mdl_info.model} not supported in Stastsmodels GLM. Supported families are:
    - \"Gaussian\" for {ModelEnum.STATSMODELS_GAUSSIAN_GLM},
    - \"Gamma\" for {ModelEnum.STATSMODELS_GAMMA_GLM},
    - \"Poisson\" for {ModelEnum.STATSMODELS_POISSON_GLM},
    - \"Tweedie\" fpr {ModelEnum.STATSMODELS_TWEEDIE_GLM}.""")
        self._set_hparams({})
        self.model = None

    def _fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame, **kwargs):
        if type(target_df) is pd.DataFrame:
            target_df = target_df[self.target_col]
        self.model = sm.GLM(target_df, features_df, family=self.family).fit(**kwargs)
        self._set_features_importances(np.abs(self.model.params))

    def _predict(self, features_df: pd.DataFrame) -> pd.Series:
        return self.model.predict(features_df)

    def get_hparams_space(self) -> Dict[str, Any]:
        return {}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        for compatibility with sklearn models (for clone method)
        """
        return dict(config=self.config)
        # return dict(config=copy.deepcopy(self.config))

    def summary(self):
        return self.model.summary()


class SklearnPoissonGLM(SklearnModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, model_class=PoissonRegressor, **kwargs)

    def get_hparams_space(self) -> Dict[str, Any]:
        return {
            'alpha': hp.loguniform('alpha', -5, 0),
            'fit_intercept': hp.choice('fit_intercept', [True, False])
        }
