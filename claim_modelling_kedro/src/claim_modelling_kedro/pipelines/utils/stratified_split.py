import logging
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_stratified_bin_labels_by_proportions(
    target: pd.Series,
    sample_weight: pd.Series,
    proportions: List[float],
    random_seed: int = 0
) -> pd.Series:
    """
    Assigns observations to bins (e.g., train/test/validation) so that the weighted sum of the target
    in each bin approximates the given proportions.

    This is useful for creating stratified splits based on a continuous target, particularly when observations
    have associated weights (e.g., exposure or policy weight).

    Args:
        target (pd.Series): A continuous target variable (e.g., average claim severity).
        sample_weight (pd.Series): Observation weights, same index as `target`.
        proportions (List[float]): A list of proportions for the bins. Must sum to 1.0.
                                   Example: [0.6, 0.2, 0.2] for train/val/test.
        random_seed (int): Random seed used for shuffling equal target values for stability.

    Returns:
        pd.Series: A Series of bin assignments (integer labels from 0 to len(proportions) - 1),
                   indexed the same as `target`.
    """
    # Sanity checks
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Proportions must sum to 1.0")

    n_bins = len(proportions)
    sample_weight = sample_weight.loc[target.index]

    # Prepare a DataFrame for sorting and computation
    df = pd.DataFrame({
        "target": target,
        "weight": sample_weight
    })
    df["weighted_target"] = df["target"] * df["weight"]

    # Shuffle observations with identical target values, then sort descending
    df = df.sample(frac=1.0, random_state=random_seed).sort_values(by="target", ascending=False)

    # Initialize containers to track current sums and assignment
    bin_sums = [0.0 for _ in range(n_bins)]  # current weighted target per bin
    weighted_target_sum = df["weighted_target"].sum()  # total weighted target sum
    bin_targets = [p * weighted_target_sum for p in proportions]  # target sum per bin
    bin_assignments = [[] for _ in range(n_bins)]

    # Assign each observation to the bin currently furthest below its target sum
    for idx, row in df.iterrows():
        residuals = [current_sum / target_sum for target_sum, current_sum in zip(bin_targets, bin_sums)]
        best_bin = int(np.argmin(residuals))
        bin_assignments[best_bin].append(idx)
        bin_sums[best_bin] += row["weighted_target"]

    # Create output Series with assigned bin labels
    bin_labels = pd.Series(index=target.index, dtype=int)
    for bin_id, indices in enumerate(bin_assignments):
        bin_labels.loc[indices] = bin_id

    return bin_labels


def _get_size_proportion(sample_size: Union[int, float], data_size: int) -> float:
    if isinstance(sample_size, float):
        return sample_size
    elif isinstance(sample_size, int):
        return sample_size / data_size
    else:
        raise ValueError("Size must be float or int")


def get_stratified_train_calib_test_split_keys(
    target_df: pd.DataFrame,
    stratify_target_col: str,
    test_size: Union[int, float],
    shuffle: bool = True,
    random_seed: int = 0,
    verbose: bool = True,
    calib_size: Union[int, float] = None,
    sample_weight: Optional[pd.Series] = None
) -> Tuple[pd.Index, Optional[pd.Index], pd.Index]:
    """
    Splits the dataset into stratified train, calibration, and test sets using bin proportions.

    Args:
        target_df (pd.DataFrame): The DataFrame containing the target column.
        stratify_target_col (str): Column used for stratification (e.g., severity).
        test_size (Union[int, float]): Size of the test set (float = percentage, int = absolute).
        shuffle (bool): Shuffle observations before sorting. Default is True.
        random_seed (int): Seed for reproducibility.
        verbose (bool): Whether to log messages.
        calib_size (Union[int, float], optional): Size of the calibration set (float or int).
        sample_weight (pd.Series, optional): Optional observation weights.

    Returns:
        Tuple[pd.Index, Optional[pd.Index], pd.Index]: Train, calibration (or None), and test indices.
    """
    # Total number of observations
    n = len(target_df)

    # Compute proportions from sizes


    p_test = _get_size_proportion(test_size, n)
    p_calib = _get_size_proportion(calib_size, n) if calib_size is not None else 0.0
    p_train = 1.0 - p_test - p_calib

    if p_train < 0:
        raise ValueError("Combined test_size and calib_size exceed 1.0")

    proportions = [p_train, p_calib, p_test] if calib_size is not None else [p_train, p_test]

    # Run stratified binning
    target = target_df[stratify_target_col]
    weights = sample_weight if sample_weight is not None else pd.Series(1, index=target.index)

    bin_labels = _get_stratified_bin_labels_by_proportions(target, weights, proportions, random_seed=random_seed)

    # Extract index masks
    train_keys = bin_labels[bin_labels == 0].index
    calib_keys = bin_labels[bin_labels == 1].index if calib_size is not None else None
    test_keys = bin_labels[bin_labels == 2].index if calib_size is not None else bin_labels[bin_labels == 1].index

    if verbose:
        msg = (
            f"Stratified Train/Calib/Test Split completed with {len(target_df)} rows:\n"
            f"    – train: {len(train_keys)} ({len(train_keys) / n:.2%})"
        )
        if calib_size is not None:
            msg += f"\n    – calibration: {len(calib_keys)} ({len(calib_keys) / n:.2%})"
        msg += f"\n    – test: {len(test_keys)} ({len(test_keys) / n:.2%})"
        logger.info(msg)
        
    if shuffle:
        # Shuffle the keys to ensure randomness
        train_keys = train_keys.to_series().sample(frac=1.0, random_state=random_seed).index
        if calib_keys is not None:
            calib_keys = calib_keys.to_series().sample(frac=1.0, random_state=random_seed).index
        test_keys = test_keys.to_series().sample(frac=1.0, random_state=random_seed).index

    return train_keys, calib_keys, test_keys


def get_stratified_sample_keys(target_df: pd.DataFrame, stratify_target_col: str, size: Union[int, float],
                               shuffle: bool = True, random_seed: int = 0, verbose: bool = True,
                               sample_weight: Optional[pd.Series] = None) -> pd.Index:
    """
    Samples a stratified subset of keys from the target DataFrame based on the specified size.
    """
    # Total number of observations
    n = len(target_df)
    size = _get_size_proportion(size, n)
    _, _, keys = get_stratified_train_calib_test_split_keys(target_df, stratify_target_col, test_size=size,
                                                            shuffle=shuffle, random_seed=random_seed, verbose=verbose,
                                                            sample_weight=sample_weight)
    if shuffle:
        keys = keys.to_series().sample(frac=1.0, random_state=random_seed).index
    return keys



def get_stratified_train_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str,
                                         test_size: Union[int, float], shuffle: bool = True, random_seed: int = 0,
                                         verbose: bool = True, sample_weight: Optional[pd.Series] = None
                                         ) -> Tuple[pd.Index, pd.Index]:
    train_keys, _, test_keys = get_stratified_train_calib_test_split_keys(target_df, stratify_target_col,
                                                                          test_size=test_size,
                                                                          shuffle=shuffle,
                                                                          random_seed=random_seed,
                                                                          verbose=verbose,
                                                                          sample_weight=sample_weight)
    return train_keys, test_keys
