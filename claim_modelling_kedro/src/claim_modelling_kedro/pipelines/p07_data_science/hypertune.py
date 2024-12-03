import logging
import time
from datetime import timedelta
from typing import Dict, Any, List

import hyperopt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, Trials

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.ds_config import HyperoptAlgoEnum
from claim_modelling_kedro.pipelines.p07_data_science.model import PredictiveModel
from claim_modelling_kedro.pipelines.utils.metrics import Metric
from claim_modelling_kedro.pipelines.utils.stratified_split import get_stratified_train_test_split_keys
from claim_modelling_kedro.pipelines.utils.stratified_cv_split import get_stratified_train_test_cv
from claim_modelling_kedro.pipelines.utils.utils import get_partition, get_mlflow_run_id_for_partition, get_class_from_path, \
    convert_np_to_native

logger = logging.getLogger(__name__)

# Define the artifact path for hyperparameters tuning
_hypertune_artifact_path = "hyperopt"


def objective(hparams: Dict[str, any], round_counter: List[int], config: Config, model: PredictiveModel, metric: Metric,
              selected_sample_features_df: pd.DataFrame,
              sample_target_df: pd.DataFrame) -> float:
    round_counter[0] += 1
    round = round_counter[0]
    hparams = convert_np_to_native(hparams)
    msg = f"Hyperopt round {round}. Hyperparameters:':\n"
    for param, value in hparams.items():
        msg += f"    - {param}: {value}\n"
    msg += f"Fitting the predictive model(s) for hyperopt round {round}..."
    logger.info(msg)
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

    scores = []

    for fold in train_keys_cv.keys():
        train_keys = train_keys_cv[fold]
        val_keys = val_keys_cv[fold]
        train_features_df = selected_sample_features_df.loc[train_keys]
        val_features_df = selected_sample_features_df.loc[val_keys]
        train_target_df = sample_target_df.loc[train_keys]
        val_target_df = sample_target_df.loc[val_keys]
        # Set hyperparameters
        model.update_hparams(hparams)
        # Fit the model on the training data and measure the time
        logger.debug(f"Hyperopt fold: {fold}. Fitting the predictive model...")
        start_time = time.time()
        model.fit(train_features_df, train_target_df)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logger.debug(f"Hyperopt fold: {fold}. Fitted the predictive model. Elapsed time: {formatted_time}.")
        # Evaluate the model –on the validation data
        val_predictions_df = model.predict(val_features_df)
        score = metric.eval(val_target_df, val_predictions_df)
        scores.append(score)
        logger.debug(f"Hyperopt fold: {fold}. Evaluated the predictive model. Score: {score}.")
    # Calculate the mean score and loss
    score_mean = np.mean(scores)
    loss = -score_mean if metric.is_larger_better() else score_mean
    logger.info(f"Hyperopt round {round} – loss: {loss}, mean score: {score_mean},\nscores: {scores}. ")
    return loss


def hypertune_part(config: Config, selected_sample_features_df: pd.DataFrame,
                   sample_target_df: pd.DataFrame, save_best_hparams_in_mlfow: bool = True) -> Dict[str, Any]:
    match config.ds.hopt_algo:
        case HyperoptAlgoEnum.TPE:
            hopt_algo = hyperopt.tpe.suggest
        case HyperoptAlgoEnum.RANDOM:
            hopt_algo = hyperopt.random.suggest
        case _:
            raise ValueError(f"Hyperopt algorithm {config.ds.hopt_algo} not supported.")
    model = get_class_from_path(config.ds.model_class)(config=config)
    model.update_hparams(config.ds.model_const_hparams)
    metric = model.metric() if config.ds.hopt_metric is None else Metric.from_enum(config, config.ds.hopt_metric,
                                                                                   pred_col=config.mdl_task.prediction_col)
    hparam_space = model.get_hparams_space()
    for hparam in config.ds.hopt_excluded_params + list(config.ds.model_const_hparams.keys()):
        if hparam in hparam_space:
            hparam_space.pop(hparam)
    trials = Trials()
    round_counter = [0]
    best_hparams = fmin(
        fn=lambda hparams: objective(hparams, round_counter, config, model, metric, selected_sample_features_df, sample_target_df),
        space=hparam_space,
        algo=hopt_algo,
        max_evals=config.ds.hopt_max_evals,
        trials=trials,
        rstate=np.random.default_rng(config.ds.hopt_fmin_random_seed)
    )
    best_hparams = convert_np_to_native(best_hparams)
    best_hparams = model.get_hparams_from_hyperopt_res(best_hparams)
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
            best_hparams_part = hypertune_part(config, selected_sample_features_part_df, sample_target_part_df)
            best_hparams[part] = best_hparams_part
            msg = f"The best hyperparameters for partition '{part}':\n"
            for param, value in best_hparams_part.items():
                msg += f"    - {param}: {value}\n"
            logger.info(msg)
    if save_best_hparams_in_mlfow:
        mlflow.log_dict(best_hparams, f"{_hypertune_artifact_path}/best_hparams.yml")
    return best_hparams
