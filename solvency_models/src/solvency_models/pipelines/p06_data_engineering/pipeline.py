from kedro.pipeline import Pipeline, pipeline, node

from solvency_models.pipelines.p06_data_engineering.nodes import transform_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_features,
                inputs=["config", "sample_features_df"],
                outputs="transformed_sample_features_df",
                name="transform_features",
            ),
        ]
    )
