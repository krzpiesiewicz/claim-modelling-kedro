from kedro.pipeline import Pipeline, node, pipeline

from claim_modelling_kedro.pipelines.p05_sampling.nodes import sample


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample,
                inputs=["config", "features_df", "target_df", "train_keys", "calib_keys", "is_event"],
                outputs=["sample_keys", "sample_features_df", "sample_target_df"],
                name="sample",
            ),
        ]
    )
