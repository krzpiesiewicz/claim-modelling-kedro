from claim_modelling_kedro.pipelines.p07_data_science.selectors import ModelWrapperFeatureSelector
from claim_modelling_kedro.pipelines.p07_data_science.models import PyGLMNetGLM


class PyGLMNetFeatureSelector(ModelWrapperFeatureSelector):
    def __init__(self, config):
        super().__init__(config=config, model_class=PyGLMNetGLM)
