from sklearn.dummy import DummyRegressor

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p07_data_science.models.sklearn_model import SklearnModel
from claim_modelling_kedro.pipelines.utils.metrics.sklearn_like_metric import SklearnLikeMetric, RootMeanSquaredError
from typing import Collection, Dict, Any


class DummyMeanRegressor(SklearnModel):
    """
    A dummy regressor that predicts the mean of the target variable.
    """
    def __init__(self, config: Config, **kwargs):
        """
        Initializes the DummyMeanRegressor with a hardcoded mean strategy.

        Args:
            config (Config): Configuration object.
            **kwargs: Additional arguments (ignored).
        """
        # Hardcode the DummyRegressor with the "mean" strategy
        SklearnModel.__init__(self, config=config, model_class=DummyRegressor, **kwargs)

    def metric(self) -> SklearnLikeMetric:
        """
        Returns the evaluation metric for the model.

        Returns:
            SklearnLikeMetric: Root Mean Squared Error (RMSE) metric.
        """
        return RootMeanSquaredError(self.config, pred_col=self.pred_col)

    @classmethod
    def _sklearn_hparams_names(cls) -> Collection[str]:
        return ["strategy"]

    @classmethod
    def get_hparams_space(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_default_hparams(cls) -> Dict[str, Any]:
        return {"strategy": "mean"}
