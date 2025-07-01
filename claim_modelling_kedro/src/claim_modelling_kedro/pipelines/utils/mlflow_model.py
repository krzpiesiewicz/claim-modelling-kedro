import inspect
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd


logger = logging.getLogger(__name__)


def _has_argument(method):
    sig = inspect.signature(method)
    params = sig.parameters
    return len(params) > 1  # Exclude 'self' for instance methods


def _assert_model_has_transform_method(model):
    if not hasattr(model, "transform"):  # or not _has_argument(model.transform):
        raise ValueError("The model in MLFlowModelLogger must have a \"transform\" method "
                         "which have at least one argument.")


class MLFlowModelLogger(mlflow.pyfunc.PythonModel):
    model_name: str
    verbose: bool

    def __init__(self, model, model_name: str = "model"):
        _assert_model_has_transform_method(model)
        self.model = model
        self.model_name = model_name

    def predict(self, *args, **kwargs):
        """
        This function is only for compatibility with the MLflow API.
        It is required by mlflow.pyfunc.PythonModel.
        """
        return self.model

    def log_model(self, path: str):
        logger.info(f"Saving the {self.model_name} of type {type(self.model)} to MLFlow path {path}...")
        # Remove existing artifacts for this path from local MLflow storage if they exist
        import mlflow
        import shutil
        import os
        run = mlflow.active_run()
        if run is not None:
            artifact_uri = run.info.artifact_uri
            artifact_path = os.path.join(artifact_uri, path)
            if os.path.exists(artifact_path):
                logger.info(f"Removing existing MLflow artifact path: {artifact_path}")
                shutil.rmtree(artifact_path)
        # Test pickling the model before logging to MLflow
        import pickle
        try:
            pickle.dumps(self.model)
            logger.info(f"Model {self.model_name} is picklable.")
        except Exception as e:
            logger.error(f"Model {self.model_name} is NOT picklable: {e}")
            raise
        mlflow.pyfunc.log_model(artifact_path=path, python_model=self)
        logger.info(f"Successfully saved the {self.model_name} in {path}.")


class MLFlowModelLoader():
    model_name: str
    verbose: bool

    def __init__(self, model_name: str = "model"):
        self.model_name = model_name

    def load_model(self, path: str, run_id: str = None):
        # If no MLflow run ID is provided, use the ID of the currently active run
        if run_id is None:
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Using the active MLFlow run ID: {run_id}.")
        else:
            logger.info(f"Using the provided MLFlow run ID: {run_id}.")
        # Define the URI for the MLflow model
        model_uri = f"runs:/{run_id}/{path}"
        logger.info(f"Loading the {self.model_name} from MLFlow path {path}...")
        # Load the model wrapper from the MLflow model registry and extract the model
        wrapper = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Typ wrappera: {type(wrapper)}, dostępne metody: {dir(wrapper)}")
        model = wrapper.predict(pd.DataFrame([]))
        logger.info(f"Successfully loaded the {self.model_name} of type {type(model)}.")
        # Assert that the model has a transform method
        _assert_model_has_transform_method(model)
        return model
