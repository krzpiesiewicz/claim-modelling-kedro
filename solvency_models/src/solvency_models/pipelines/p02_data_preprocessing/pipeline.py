from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, split_features_and_targets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["config", "raw_features_and_claims_numbers_df", "raw_severities_df"],
                outputs="filtered_joined_df",
                name="preprocess_data",
            ),
            node(
                func=split_features_and_targets,
                inputs=["config", "filtered_joined_df"],
                outputs=["features_df", "targets_df"],
                name="split_features_and_targets",
            ),
        ]
    )
