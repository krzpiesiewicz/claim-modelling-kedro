import logging
import time
from datetime import timedelta
from functools import partial
from typing import Dict, Any

import hyperopt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, Trials, space_eval, STATUS_OK, STATUS_FAIL

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.ds_config import HyperoptAlgoEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric
from claim_modelling_kedro.pipelines.utils.stratified_cv_split import get_stratified_train_test_cv
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys
from claim_modelling_kedro.pipelines.utils.utils import get_partition, get_mlflow_run_id_for_partition, \
    get_class_from_path, \
    round_decimal, save_pd_dataframe_as_csv_in_mlflow

logger = logging.getLogger(__name__)

# Define the artifact path for hyperparameters tuning
_hypertune_artifact_path = "hyperopt"


def log_trials_info_to_mlflow(
        trials: Trials,
        space: Dict[str, Any],
        log_folds_metrics: bool,
        artifact_path: str
) -> None:
    """
    Create a DataFrame from the trials and log it to MLFlow.
    """
    trials_df = pd.DataFrame([{
        "trial_no": trial["tid"] + 1,
        "book_time": trial["book_time"],
        "eval_time": trial["result"].get("eval_time"),
        "status": trial["result"].get("status"),
        "loss": round_decimal(trial["result"].get("loss"), 3)
    } for trial in trials.trials])
    trials_df = pd.concat([trials_df, pd.DataFrame([{
        **{k: round_decimal(v, 3) for k, v in ((trial.get("attachments") or {}).get("metrics") or {}).items()}
    } for trial in trials.trials])], axis=1)
    trials_df = pd.concat([trials_df, pd.DataFrame([
        {**space_eval(space, {k: v[0] if isinstance(v, list) else v for k, v in trial["misc"]["vals"].items()})
         } for trial in trials.trials])], axis=1)
    save_pd_dataframe_as_csv_in_mlflow(trials_df, artifact_path, "trials.csv", index=False)

    if log_folds_metrics:
        rows = []
        for trial in trials.trials:
            for dataset in ["valid", "train"]:
                rows.append({
                    "trial_no": trial["tid"] + 1,
                    "dataset": dataset,
                    **{fold: round_decimal(score, 3) for fold, score in
                       enumerate((trial.get("attachments") or {}).get(f"{dataset}_scores") or [])},
                })
        trials_scores_df = pd.DataFrame(rows)
        save_pd_dataframe_as_csv_in_mlflow(trials_scores_df, artifact_path, "trials_folds_scores.csv", index=False)


def fit_model(hparams: Dict[str, any],
              model: PredictiveModel,
              config: Config,
              trials: Trials,
              space: Dict[str, Any],
              metric: Metric,
              selected_sample_features_df: pd.DataFrame,
              sample_target_df: pd.DataFrame,
              hyperopt_artifact_path: str) -> float:
    trial_no = len(trials.trials)
    msg = f"Hyperopt trial {trial_no}. Hyperparameters:\n"
    for param, value in hparams.items():
        msg += f"    - {param}: {value}\n"
    msg += f"Fitting the predictive model(s) for hyperopt trial {trial_no}..."
    logger.info(msg)
    log_trials_info_to_mlflow(trials, space, log_folds_metrics=config.ds.hopt_cv_enabled,
                              artifact_path=hyperopt_artifact_path)

    if config.ds.hopt_cv_enabled:
        train_keys_cv, val_keys_cv = get_stratified_train_test_cv(sample_target_df,
                                                                  stratify_target_col=config.mdl_task.target_col,
                                                                  cv_folds=config.ds.hopt_cv_folds, shuffle=True,
                                                                  random_seed=config.ds.hopt_split_random_seed,
                                                                  verbose=False)
    else:
        train_keys, test_keys = get_stratified_train_test_split_keys(sample_target_df,
                                                                     stratify_target_col=config.mdl_task.target_col,
                                                                     test_size=config.ds.hopt_split_val_size,
                                                                     shuffle=True,
                                                                     random_seed=config.ds.hopt_split_random_seed,
                                                                     verbose=False)
        train_keys_cv = {"0": train_keys}
        val_keys_cv = {"0": test_keys}

    train_scores = []
    val_scores = []
    metric_name = metric.get_short_name()
    start_time = time.time()

    try:
        for fold in train_keys_cv.keys():
            train_keys = train_keys_cv[fold]
            val_keys = val_keys_cv[fold]
            train_features_df = selected_sample_features_df.loc[train_keys]
            val_features_df = selected_sample_features_df.loc[val_keys]
            train_target_df = sample_target_df.loc[train_keys]
            val_target_df = sample_target_df.loc[val_keys]

            model.update_hparams(hparams)
            logger.debug(f"Hyperopt fold: {fold}. Fitting the predictive model...")
            model.fit(train_features_df, train_target_df)
            logger.debug(f"Hyperopt fold: {fold}. Fitted the predictive model.")

            train_predictions_df = model.predict(train_features_df)
            val_predictions_df = model.predict(val_features_df)

            train_score = metric.eval(train_target_df, train_predictions_df)
            val_score = metric.eval(val_target_df, val_predictions_df)

            train_scores.append(train_score)
            val_scores.append(val_score)

            logger.debug(
                f"Hyperopt fold: {fold}. Train score ({metric_name}): {train_score}, Validation score ({metric_name}): {val_score}.")

        train_score_mean = np.mean(train_scores)
        val_score_mean = np.mean(val_scores)
        loss = -val_score_mean if metric.is_larger_better() else val_score_mean
        if config.ds.hopt_cv_enabled:
            train_score_std = np.std(train_scores)
            val_score_std = np.std(val_scores)
        status = STATUS_OK
    except Exception as e:
        logger.error(f"Error during hyperopt trial {trial_no}: {e}")
        train_score_mean = np.nan
        val_score_mean = np.nan
        train_score_std = np.nan
        val_score_std = np.nan
        loss = np.nan
        status = STATUS_FAIL

    eval_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=eval_time))
    msg = f"Hyperopt trial {trial_no} – eval_time: {formatted_time} - status: {status} - loss: {loss}"
    metrics_results = {
        f"{metric_name}_valid": val_score_mean,
        f"{metric_name}_train": train_score_mean,
    }
    if config.ds.hopt_cv_enabled:
        metrics_results.update({
            f"{metric_name}_std_valid": val_score_std,
            f"{metric_name}_std_train": train_score_std,
        })
        scores_df = pd.DataFrame({
            f"valid": map(partial(round_decimal, significant_digits=3), val_scores),
            f"train": map(partial(round_decimal, significant_digits=3), train_scores)
        })
        scores_df.index.name = "fold"
        msg = (
            f"{msg}\n"
            f"{metric_name}_valid: {round_decimal(val_score_mean, 3)} ± {round_decimal(val_score_std, 3)} (std)\n"
            f"{metric_name}_train: {round_decimal(train_score_mean, 3)} ± {round_decimal(train_score_std, 3)} (std)\n"
            f"{metric_name}_scores:\n{scores_df}"
        )
    else:
        msg = (
            f"{msg}\n"
            f"{metric_name}_valid: {round_decimal(val_score_mean, 3)}\n"
            f"{metric_name}_train: {round_decimal(train_score_mean, 3)}"
        )
    logger.info(msg)

    attachments = {
        "metrics": metrics_results,
        "train_scores": train_scores,
        "valid_scores": val_scores,
    }
    trial_result = {
        "trial_no": trial_no,
        "book_time": time.gmtime(start_time),
        "eval_time": formatted_time,
        "status": status,
        "loss": loss,
    }
    trials.trials[-1]["attachments"] = attachments
    trials.trials[-1]["result"] = trial_result
    log_trials_info_to_mlflow(trials, space, log_folds_metrics=config.ds.hopt_cv_enabled,
                              artifact_path=hyperopt_artifact_path)
    return {
        **trial_result,
        "attachments": attachments
    }


def hypertune_part(config: Config, selected_sample_features_df: pd.DataFrame,
                   sample_target_df: pd.DataFrame, hyperopt_artifact_path: str,
                   save_best_hparams_in_mlfow: bool = True) -> Dict[str, Any]:
    match config.ds.hopt_algo:
        case HyperoptAlgoEnum.TPE:
            hopt_algo = hyperopt.tpe.suggest
        case HyperoptAlgoEnum.RANDOM:
            hopt_algo = hyperopt.random.suggest
        case _:
            raise ValueError(f"Hyperopt algorithm {config.ds.hopt_algo} not supported.")
    model = get_class_from_path(config.ds.model_class)(config=config, target_col=config.mdl_task.target_col,
                                                       pred_col=config.mdl_task.prediction_col)
    model.update_hparams(config.ds.model_const_hparams)
    metric = model.metric() if config.ds.hopt_metric is None else Metric.from_enum(config, config.ds.hopt_metric,
                                                                                   pred_col=config.mdl_task.prediction_col)
    hparam_space = model.get_hparams_space()
    for hparam in config.ds.hopt_excluded_params + list(config.ds.model_const_hparams.keys()):
        if hparam in hparam_space:
            hparam_space.pop(hparam)
    if hparam_space is None or len(hparam_space) == 0:
        logger.warning(
            "No hyperparameters to tune. Please check the hyperparameters space. Returning an empty dict as best_hparams.")
        return {}

    trials = Trials()
    objective = partial(fit_model, config=config, model=model, trials=trials, space=hparam_space, metric=metric,
                        selected_sample_features_df=selected_sample_features_df,
                        sample_target_df=sample_target_df, hyperopt_artifact_path=hyperopt_artifact_path)
    hp_assignment = fmin(
        fn=objective,
        space=hparam_space,
        algo=hopt_algo,
        max_evals=config.ds.hopt_max_evals,
        trials=trials,
        rstate=np.random.default_rng(config.ds.hopt_fmin_random_seed)
    )
    best_hparams = space_eval(hparam_space, hp_assignment)

    if save_best_hparams_in_mlfow:
        mlflow.log_dict(best_hparams, f"{_hypertune_artifact_path}/best_hparams.yml")
    return best_hparams


def hypertune(config: Config, selected_sample_features_df: Dict[str, pd.DataFrame],
              sample_target_df: Dict[str, pd.DataFrame],
              save_best_hparams_in_mlfow: bool = True) -> Dict[str, Dict[str, Any]]:
    logger.info(f"Tuning the hyper parameters of the predictive model {config.ds.model_class}...")
    best_hparams = {}
    for part in selected_sample_features_df.keys():
        selected_sample_features_part_df = get_partition(selected_sample_features_df, part)
        sample_target_part_df = get_partition(sample_target_df, part)
        mlflow_subrun_id = get_mlflow_run_id_for_partition(config, part)
        logger.info(
            f"Tuning the hyper parameters of the predictive model on partition '{part}' of the sample dataset...")
        with mlflow.start_run(run_id=mlflow_subrun_id, nested=True):
            best_hparams_part = hypertune_part(config, selected_sample_features_part_df, sample_target_part_df,
                                               hyperopt_artifact_path=f"{_hypertune_artifact_path}/{part}")
            best_hparams[part] = best_hparams_part
            msg = f"The best hyperparameters for partition '{part}':\n"
            for param, value in best_hparams_part.items():
                msg += f"    - {param}: {value}\n"
            logger.info(msg)
    if save_best_hparams_in_mlfow:
        mlflow.log_dict(best_hparams, f"{_hypertune_artifact_path}/best_hparams.yml")
    return best_hparams
