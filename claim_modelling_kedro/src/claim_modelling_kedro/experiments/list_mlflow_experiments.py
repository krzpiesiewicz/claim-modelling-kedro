import os
import yaml
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def get_tracking_uri_from_config(config_path="claim_modelling_kedro/conf/local/mlflow.yml") -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["server"]["mlflow_tracking_uri"]


def list_experiments():
    tracking_uri = get_tracking_uri_from_config()
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    client = MlflowClient()
    experiments = client.search_experiments(view_type=ViewType.ALL)

    print(f"\n📋 Found {len(experiments)} experiment(s):\n")
    for exp in experiments:
        print(f"🧪 Name: '{exp.name}'\n   ID: {exp.experiment_id}\n   Status: {exp.lifecycle_stage}\n")


if __name__ == "__main__":
    list_experiments()
