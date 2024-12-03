from kedro.pipeline import Pipeline, node, pipeline

from claim_modelling_kedro.pipelines.p03_define_modeling_task.nodes import create_modeling_task


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_modeling_task,
                inputs=["config", "targets_df"],
                outputs=["target_df", "is_event"],
                name="create_modeling_task",
            ),
        ]
    )
