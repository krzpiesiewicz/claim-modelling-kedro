#!/usr/bin/env python3
"""
CLI tool to manage MLflow experiments
-------------------------------------

Supported actions:
  • delete   – soft-delete (marks experiment as *deleted*)
  • restore  – restore a soft-deleted experiment
  • purge    – PERMANENTLY delete (SQL back-end + MLflow ≥ 2.7)

Tracking URI is read from:
   claim_modelling_kedro/conf/local/mlflow.yml
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

CONFIG_PATH = Path("claim_modelling_kedro/conf/local/mlflow.yml")


def load_tracking_uri() -> str:
    try:
        with CONFIG_PATH.open("r") as f:
            cfg = yaml.safe_load(f)
        return cfg["server"]["mlflow_tracking_uri"]
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Could not load tracking URI from {CONFIG_PATH}: {e}", file=sys.stderr)
        sys.exit(1)


def get_experiment(client: MlflowClient, name: str | None, exp_id: str | None):
    if name:
        return client.get_experiment_by_name(name)
    if exp_id:
        return client.get_experiment(exp_id)
    return None


def build_argparser() -> argparse.ArgumentParser:
    examples = """\
Examples
--------

# Soft-delete experiment by name
manage_mlflow_experiment.py delete --name sev_001_dummy_mean_regressor

# Restore experiment by ID
manage_mlflow_experiment.py restore --id 12

# Permanently delete (⚠ requires SQL backend + MLflow ≥ 2.7)
manage_mlflow_experiment.py purge --name sev_001_dummy_mean_regressor
"""
    parser = argparse.ArgumentParser(
        prog="manage_mlflow_experiment.py",
        description="Soft-delete, restore or permanently purge MLflow experiments.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for action in ("delete", "restore", "purge"):
        sub = subparsers.add_parser(
            action,
            help=f"{action} an experiment",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        mg = sub.add_mutually_exclusive_group(required=True)
        mg.add_argument("--name", help="Experiment name")
        mg.add_argument("--id", help="Experiment ID")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    tracking_uri = load_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = get_experiment(client, args.name, args.id)
    if exp is None:
        print("❌ Experiment not found.", file=sys.stderr)
        sys.exit(1)

    if args.command == "delete":
        if exp.lifecycle_stage == "deleted":
            print(f"⚠️  Experiment '{exp.name}' is already soft-deleted.")
        else:
            client.delete_experiment(exp.experiment_id)
            print(f"🗑️  Soft-deleted experiment '{exp.name}' (ID {exp.experiment_id}).")

    elif args.command == "restore":
        if exp.lifecycle_stage == "active":
            print(f"✅ Experiment '{exp.name}' is already active.")
        else:
            client.restore_experiment(exp.experiment_id)
            print(f"♻️  Restored experiment '{exp.name}' (ID {exp.experiment_id}).")

    elif args.command == "purge":
        try:
            confirm = input(
                f"⚠️  Permanently delete experiment '{exp.name}' (ID {exp.experiment_id})? [y/N]: "
            )
            if confirm.strip().lower() != "y":
                print("Aborted.")
                sys.exit(0)
            client.delete_experiment(exp.experiment_id, permanent=True)
            print(f"💥 Permanently deleted experiment '{exp.name}'.")
        except TypeError:
            print(
                "❌ Permanent delete not supported by this MLflow version or backend.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except MlflowException as e:
        print(f"❌ MLflowException: {e}", file=sys.stderr)
        sys.exit(1)
