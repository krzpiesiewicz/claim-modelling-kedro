from kedro.pipeline import Pipeline, node, pipeline

from solvency_models.pipelines.p04_sampling.nodes import sample


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample,
                inputs=["config", "train_idx", "target_df"],
                outputs="sampled_policies",
                name="sample",
            ),
        ]
    )
