from kedro.pipeline import Pipeline, node, pipeline


def dummy_function(*args, **kwargs):
    pass

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dummy_function,
                inputs="parameters",
                outputs=None,
                name="dummy_node",
            ),
        ]
    )
