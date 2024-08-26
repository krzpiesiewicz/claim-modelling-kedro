from kedro.pipeline import Pipeline, pipeline, node

from solvency_models.pipelines.p06_data_engineering.nodes import fit_features_transformer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_features_transformer,
                inputs=["config", "sample_features_df"],
                outputs="transformed_sample_features_df",
                name="fit_features_transformer",
            ),
        ]
    )
