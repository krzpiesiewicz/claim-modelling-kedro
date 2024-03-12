from kedro.pipeline import Pipeline, node, pipeline

from .nodes import process_parameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_parameters,
                inputs="parameters",
                outputs="config",
                name="process_parameters",
            ),
        ]
    )
