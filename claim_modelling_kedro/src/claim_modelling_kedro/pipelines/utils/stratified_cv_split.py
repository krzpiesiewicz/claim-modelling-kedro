import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_balanced_folds_labels_by_weighted_target(
        target: pd.Series,
        weights: pd.Series,
        n_folds: int,
        random_seed: int = 0
) -> pd.Series:
    """
    Przypisuje obserwacje do foldów tak, aby sumy weighted_target były możliwie równe.

    Args:
        target (pd.Series): Ciągła zmienna targetowa (np. średnia szkoda).
        weights (pd.Series): Wagi obserwacji.
        n_folds (int): Liczba foldów.
        random_seed (int): Losowy seed do mieszania identycznych wartości.

    Returns:
        pd.Series: Przypisanie do foldów (wartości od 0 do n_folds-1).
    """
    assert (target.index == weights.index).all(), "target i weights muszą mieć identyczny index"

    df = pd.DataFrame({
        "target": target,
        "weight": weights
    })
    df["weighted_target"] = df["target"] * df["weight"]

    # Shuffle w ramach identycznych wartości targetu (opcjonalne, dla stabilności)
    df = df.sample(frac=1, random_state=random_seed).sort_values(by="target", ascending=False)

    # Inicjalizacja foldów
    fold_sums = [0.0 for _ in range(n_folds)]
    fold_assignments = [[] for _ in range(n_folds)]

    for idx, row in df.iterrows():
        # Znajdź fold o najmniejszej sumie weighted target
        min_fold = np.argmin(fold_sums)
        fold_assignments[min_fold].append(idx)
        fold_sums[min_fold] += row["weighted_target"]

    # Zbuduj Series z przypisaniem do foldów
    folds_labels = pd.Series(index=target.index, dtype=int)
    for fold_id, indices in enumerate(fold_assignments):
        folds_labels.loc[indices] = fold_id

    return folds_labels


def _get_folds_labels(target: pd.Series, cv_folds: int,
                      shuffle: bool = True, random_seed: int = 0) -> pd.Series:
    n_samples = len(target)
    n_buckets = n_samples // cv_folds + 1
    fold_labels = np.tile(np.arange(cv_folds), n_buckets)[:n_samples]
    if shuffle:
        rng = np.random.default_rng(random_seed)
        for i in range(0, n_samples, cv_folds):
            rng.shuffle(fold_labels[i: min(i + cv_folds, n_samples)])
    return pd.Series(fold_labels, index=target.index, dtype=int)


def _get_stratified_train_calib_test_cv(target_df: pd.DataFrame, stratify_target_col: str, cv_folds: int,
                                        calib_set_enabled: bool = False, shared_train_calib_set: bool = False,
                                        sample_weight: pd.Series = None, balance_sample_weights: bool = True,
                                        shuffle: bool = True, random_seed: int = 0,
                                        verbose: bool = False, cv_type: str = 'train_test',
                                        cv_parts_names: list = None):
    """
    Performs stratified cross-validation splits.

    Args:
        target_df (pd.DataFrame): The DataFrame containing the target column.
        stratify_target_col (str): The column in target_df to stratify on (e.g., a continuous variable like frequency/severity).
        cv_folds (int): The number of cross-validation folds.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        random_seed (int): The random seed for reproducibility. Default is 0.
        verbose (bool): Whether to print information about the splits. Default is False.
        cv_type (str): The type of cross-validation to perform.
                       'train_test' for standard CV (train set: cv_folds-1, test set: 1 fold).
                       'train_calib_test' for calibration CV (train set: cv_folds-2, calib set: 1 fold, test set: 1 fold).

    Returns:
        Tuple[Dict[str, pd.Index], Dict[str, pd.Index], Dict[str, pd.Index]]: Dictionaries containing train, calib, and test sets for each fold.
    """

    train_keys_cv = {}
    calib_keys_cv = {}
    test_keys_cv = {}

    n_samples = target_df.shape[0]

    # Create folds_labels based on the stratify_target_col
    if balance_sample_weights:
        if sample_weight is None:
            sample_weight = pd.Series(1.0, index=target_df.index, dtype=float)
        sorted_target_df = target_df.assign(_weight=sample_weight).sort_values(
            by=[stratify_target_col, "_weight"], ascending=[False, False]
        )
        folds_labels = _get_balanced_folds_labels_by_weighted_target(
            target=sorted_target_df[stratify_target_col],
            weights=sorted_target_df["_weight"],
            n_folds=cv_folds,
            random_seed=random_seed
        )
    else:
        sorted_target_df = target_df.sort_values(by=stratify_target_col, ascending=False)
        folds_labels = _get_folds_labels(sorted_target_df, cv_folds=cv_folds, shuffle=shuffle,
                                         random_seed=random_seed)

    # Cross-validation splitting based on cv_type
    if cv_type == 'train_test':
        # Standard cross-validation (cv_folds-1 train blocks, 1 test block)
        for fold in range(cv_folds):
            # Test set is the current fold
            test_mask = (folds_labels == fold)
            test_keys = sorted_target_df[test_mask].index

            # Training set is all other blocks except the current fold
            train_mask = ~test_mask
            train_keys = sorted_target_df[train_mask].index

            part = cv_parts_names[fold] if cv_parts_names is not None else str(fold)
            train_keys_cv[part] = train_keys
            test_keys_cv[part] = test_keys

    elif cv_type == 'train_calib_test':
        # Calibration cross-validation (cv_folds-2 train blocks, 1 calibration block, 1 test block)
        for fold in range(cv_folds):
            # For each fold, use one block for calibration and one for test
            test_fold = fold
            # Test set is the current fold
            test_mask = (folds_labels == test_fold)
            test_keys = sorted_target_df[test_mask].index

            if calib_set_enabled:
                if shared_train_calib_set:
                    # If shared_train_calib_set is True, use the same train set for both train and calibration
                    # Train set: all other folds except the test fold
                    train_mask = ~test_mask
                    train_keys = sorted_target_df[train_mask].index
                    calib_keys = train_keys
                else:
                    calib_fold = (fold + 1) % cv_folds  # The calibration fold is the next one (cyclic shift)
                    # Calibration set is the next fold
                    calib_mask = (folds_labels == calib_fold)
                    calib_keys = sorted_target_df[calib_mask].index

                    # Train set: all the other folds except the test and calibration folds
                    train_mask = ~(test_mask | calib_mask)
                    train_keys = sorted_target_df[train_mask].index
            else: # calib_set_enabled is False
                # If calibration set is not enabled, calib set: empty nad train set: all other folds except the test fold
                train_mask = ~test_mask
                train_keys = sorted_target_df[train_mask].index
                calib_keys = pd.Index([])

            # Store the indices in dictionaries
            part = cv_parts_names[fold] if cv_parts_names is not None else str(fold)
            train_keys_cv[part] = train_keys
            calib_keys_cv[part] = calib_keys
            test_keys_cv[part] = test_keys

    # Step 4: Logging (if verbose mode is enabled)
    if verbose:
        avg_bin_size = n_samples / cv_folds
        if cv_type == 'train_test':
            msg = (f"Stratified CV (Train/Test). Split into {cv_folds} folds. Average bin size: {avg_bin_size:.2f}. "
                   f"The sizes of the folds:")
            for fold in train_keys_cv.keys():
                msg += (f"\n    - fold {fold}: train: {len(train_keys_cv[fold])} samples, "
                        f"test: {len(test_keys_cv[fold])} samples "
                        f"({len(test_keys_cv[fold]) / len(target_df):.2%})")
        elif cv_type == 'train_calib_test':
            msg = (
                f"Stratified CV (Train/Calib/Test). Split into {cv_folds} partitions. Average bin size: {avg_bin_size:.2f}. "
                f"The sizes of the partitions:")
            for key in train_keys_cv.keys():
                msg += (f"\n    - {key}: train: {len(train_keys_cv[key])} samples, "
                        f"calibration: {len(calib_keys_cv[key])} samples, "
                        f"test: {len(test_keys_cv[key])} samples "
                        f"({len(test_keys_cv[key]) / len(target_df):.2%})")
        logger.info(msg)

    # Return appropriate dictionaries based on the cross-validation type
    if cv_type == 'train_test':
        return train_keys_cv, test_keys_cv
    elif cv_type == 'train_calib_test':
        return train_keys_cv, calib_keys_cv, test_keys_cv


def get_stratified_train_calib_test_cv(target_df: pd.DataFrame, stratify_target_col: str, cv_folds: int,
                                       calib_set_enabled: bool, shared_train_calib_set: bool,
                                       cv_parts_names=None, sample_weight: pd.Series = None,
                                       shuffle: bool = True, random_seed: int = 0, verbose: bool = False) -> Tuple[
    Dict[str, pd.Index], Dict[str, pd.Index], Dict[str, pd.Index]]:
    return _get_stratified_train_calib_test_cv(target_df, stratify_target_col, cv_folds=cv_folds,
                                               calib_set_enabled=calib_set_enabled,
                                               shared_train_calib_set=shared_train_calib_set,
                                               sample_weight=sample_weight, shuffle=shuffle, random_seed=random_seed,
                                               verbose=verbose,
                                               cv_parts_names=cv_parts_names, cv_type='train_calib_test')


def get_stratified_train_test_cv(target_df: pd.DataFrame, stratify_target_col: str, cv_folds: int,
                                 cv_parts_names=None, sample_weight: pd.Series = None,
                                 shuffle: bool = True, random_seed: int = 0, verbose: bool = False) -> Tuple[
    Dict[str, pd.Index], Dict[str, pd.Index]]:
    return _get_stratified_train_calib_test_cv(target_df, stratify_target_col, cv_folds=cv_folds,
                                               sample_weight=sample_weight, shuffle=shuffle, random_seed=random_seed,
                                               verbose=verbose,
                                               cv_parts_names=cv_parts_names, cv_type='train_test')
