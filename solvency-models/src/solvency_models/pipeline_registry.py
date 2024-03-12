"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from solvency_models.pipelines import (
    p01_init, p02_data_preprocessing, p03_define_modeling_task, p04_sampling
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    init_pipeline = p01_init.create_pipeline()
    data_processing_pipeline = p02_data_preprocessing.create_pipeline()
    def_modeling_task_pipeline = p03_define_modeling_task.create_pipeline()
    sampling_pipeline = p04_sampling.create_pipeline()
    pipelines = dict()
    pipelines["init"] = init_pipeline
    pipelines["dp"] = init_pipeline + data_processing_pipeline
    pipelines["def_task"] = init_pipeline + def_modeling_task_pipeline
    pipelines["smpl"] = init_pipeline + def_modeling_task_pipeline + sampling_pipeline
    pipelines["all"] = init_pipeline + data_processing_pipeline + def_modeling_task_pipeline + sampling_pipeline
    pipelines["__default__"] = pipelines["all"]
    return pipelines
