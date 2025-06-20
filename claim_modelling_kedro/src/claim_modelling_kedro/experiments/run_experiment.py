#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import time
from datetime import timedelta
from typing import Tuple

from claim_modelling_kedro.experiments import (
    copy_experiment_run_configs,
    run_kedro_pipeline,
)
from claim_modelling_kedro.experiments.setup_logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ #
# 🕒 Helper: pretty-format seconds as “Xd HH:MM:SS”
# ------------------------------------------------------------------------------ #
def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    days = td.days
    hrs, rem = divmod(td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hrs:02}:{mins:02}:{secs:02}"
    return f"{hrs:02}:{mins:02}:{secs:02}"


def copy_config_and_run_experiment(
    base_dir: str,
    experiment_dir: str,
    experiment_name: str,
    run_name: str,
    pipeline: str,
) -> Tuple[int, float]:
    copy_experiment_run_configs(experiment_dir, run_name)
    os.chdir(base_dir)
    t0 = time.time()
    exit_code = run_kedro_pipeline(experiment_name, run_name, pipeline)
    elapsed = time.time() - t0
    return exit_code, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run an experiment for different pipelines.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("first_pipeline", type=str, help="Pipeline to run for the first run")
    parser.add_argument("--other-pipeline", type=str, help="Pipeline for all other runs", default=None)

    # Mutually exclusive: choose runs explicitly or start from a run
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-name", type=str, nargs="+", help="Run names to execute")
    group.add_argument("--from-run-name", type=str, help="Start from this run (inclusive)")

    args = parser.parse_args()
    experiment_name = args.experiment_name
    first_pipeline = args.first_pipeline
    other_pipeline = args.other_pipeline or first_pipeline

    base_dir = os.path.abspath(os.getcwd())
    experiment_dir = os.path.join(base_dir, "experiments", experiment_name)
    runs_dir = os.path.join(experiment_dir, "runs")

    run_names = sorted(
        d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))
    )
    run_names = [r for r in run_names if r != "dummy"]

    if not run_names:
        logger.error(f"No run directories found in {runs_dir}")
        return

    # -------- select runs ---------- #
    if args.run_name:
        invalid = [r for r in args.run_name if r not in run_names]
        if invalid:
            logger.error(f"Run(s) not found: {', '.join(invalid)}")
            return
        selected = args.run_name
    elif args.from_run_name:
        if args.from_run_name not in run_names:
            logger.error(f"Run '{args.from_run_name}' not found in {runs_dir}")
            return
        selected = run_names[run_names.index(args.from_run_name) :]
    else:
        selected = run_names

    # -------- execution loop -------- #
    total = len(selected)
    run_times = []
    t_global0 = time.time()

    logger.info(f"Running {total} run(s) for experiment '{experiment_name}':")
    logger.info("   • " + "\n   • ".join(selected))

    for idx, run_name in enumerate(selected, start=1):
        pipeline = first_pipeline if idx == 1 else other_pipeline
        logger.info(f"▶️  Run {idx}/{total}: '{run_name}'  | pipeline={pipeline}")

        exit_code, dur = copy_config_and_run_experiment(base_dir, experiment_dir, experiment_name, run_name, pipeline)
        if exit_code == 130:
            msg = f"⛔ Run {run_name} interrupted by user (exit code 130)."
            logger.warning(msg)
            subprocess.run(
                [
                    "notify-send",
                    "-a", "run_experiment.sh",
                    "-u", "critical",
                    "-t", "0",
                    f"Run interrupted – {experiment_name}",
                    f"Run: {run_name}\nUser interruption (Ctrl+C).\nExiting.",
                ],
                check=False,
            )
            break  # Stop the whole script immediately

        run_times.append(dur)

        mean_time = sum(run_times) / len(run_times) if run_times else 0.0
        remaining = total - idx
        eta = mean_time * remaining
        elapsed = time.time() - t_global0

        status = "✅ success" if exit_code == 0 else f"❌ failed (exit code {exit_code})"
        logger.info(
            f"✔ Run {idx}/{total}: {run_name} completed with status: {status} "
            f"• mean={fmt_duration(mean_time)} "
            f"• ETA={fmt_duration(eta)} "
            f"• elapsed={fmt_duration(elapsed)}"
        )

        subprocess.run(
            [
                "notify-send",
                "-a", "run_experiment.sh",
                "-u", "normal",
                "-t", "0",
                f"Completed run {idx}/{total} – {experiment_name}",
                f"Run: {run_name}\n"
                f"Status: {status}\n"
                f"Progress: {idx}/{total} ({(idx / total):.2%})\n"
                f"Mean: {fmt_duration(sum(run_times) / len(run_times))} | "
                f"ETA: {fmt_duration((total - idx) * (sum(run_times) / len(run_times)))} | "
                f"Elapsed: {fmt_duration(time.time() - t_global0)}",
            ],
            check=False,
        )

    # -------- summary notification -------- #
    summary_elapsed = fmt_duration(time.time() - t_global0)
    if args.run_name:
        k = len(args.run_name)
        if k > 6:
            abr = ", ".join(args.run_name[:3] + ["..."] + args.run_name[-3:])
            summary = f"Completed {k} selected runs: {abr}"
        else:
            summary = f"Completed runs: {', '.join(args.run_name)}"
    elif args.from_run_name:
        first_idx = run_names.index(selected[0]) + 1
        last_idx = run_names.index(selected[-1]) + 1
        summary = f"Completed runs {first_idx}–{last_idx}"
    else:
        summary = f"Completed all {total} runs"

    subprocess.run(
        [
            "notify-send",
            "-a", "run_experiment.sh",
            "-u", "normal",
            "-t", "0",
            f"Finished experiment: {experiment_name}",
            f"{summary}\nTotal time: {summary_elapsed}",
        ],
        check=False,
    )

    logger.info(f"✅ All runs finished. Total time: {summary_elapsed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
