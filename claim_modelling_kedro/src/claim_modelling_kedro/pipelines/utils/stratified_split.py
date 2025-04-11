import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_stratified_sample_keys(target_df: pd.DataFrame, stratify_target_col: str, size: Union[int, float],
                               shuffle: bool = True, random_seed: int = 0, verbose: bool = True) -> pd.Index:
    """
    Performs a stratified sampling by dividing the data into buckets and selecting one observation from each bucket.

    Args:
        target_df (pd.DataFrame): The DataFrame containing the data to sample from.
        stratify_target_col (str): The column to stratify on.
        size (Union[int, float]): The number of buckets (int) or fraction of the dataset (float) to sample.
        shuffle (bool): Whether to shuffle the data within each bucket before sampling. Default is True.
        random_seed (int): The random seed for reproducibility. Default is 0.
        verbose (bool): Whether to print information about the sampling process. Default is True.

    Returns:
        pd.Index: A Index containing the stratified sample keys.
    """
    # Ensure size is valid
    if isinstance(size, float):
        if not (0 < size <= 1):
            raise ValueError("If size is a float, it must be in the range (0, 1].")
        n_buckets = int(len(target_df) * size)
    elif isinstance(size, int):
        if size <= 0 or size > len(target_df):
            raise ValueError("If size is an int, it must be in the range [1, len(target_df)].")
        n_buckets = size
    else:
        raise TypeError("Size must be either an int or a float.")

    # Sort the DataFrame by the stratify_target_col
    sorted_df = target_df.sort_values(by=stratify_target_col, ascending=True)

    # Divide the sorted DataFrame into buckets
    bucket_size = len(sorted_df) // n_buckets
    buckets = [sorted_df.iloc[i * bucket_size:(i + 1) * bucket_size] for i in range(n_buckets)]

    # Handle any remaining samples (last bucket may be larger)
    if len(sorted_df) % n_buckets != 0:
        buckets[-1] = pd.concat([buckets[-1], sorted_df.iloc[n_buckets * bucket_size:]])

    # Sample one observation from each bucket
    sampled_indices = []
    rng = np.random.default_rng(random_seed)
    for bucket in buckets:
        if shuffle:
            sampled_indices.append(rng.choice(bucket.index, size=1, replace=False)[0])
        else:
            sampled_indices.append(bucket.index[0])  # Take the first observation if no shuffle

    if verbose:
        print(f"Stratified sampling completed. {len(sampled_indices)} samples selected from {n_buckets} buckets.")

    return sampled_indices


def get_stratified_train_calib_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str,
                                               test_size: Union[int, float], shuffle: bool = True,
                                               random_seed: int = 0, verbose: bool = True,
                                               calib_size: Union[int, float] = None) -> Tuple[
    pd.Index, pd.Index, pd.Index]:
    """
    Splits the dataset into stratified train, calibration, and test sets.

    Args:
        target_df (pd.DataFrame): The DataFrame containing the target column.
        stratify_target_col (str): The column to stratify on.
        test_size (Union[int, float]): The size of the test set (int for absolute, float for fraction).
        shuffle (bool): Whether to shuffle the data before sampling. Default is True.
        random_seed (int): The random seed for reproducibility. Default is 0.
        verbose (bool): Whether to print information about the splits. Default is True.
        calib_size (Union[int, float], optional): The size of the calibration set (int for absolute, float for fraction).

    Returns:
        Tuple[pd.Index, pd.Index, pd.Index]: Indices for the train, calibration, and test sets.
    """
    # Calculate the number of test samples
    if isinstance(test_size, float):
        n_test = int(target_df.shape[0] * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be either an int or a float.")

    # Get test keys
    test_keys = get_stratified_sample_keys(target_df, stratify_target_col, size=n_test,
                                           shuffle=shuffle, random_seed=random_seed, verbose=verbose)

    # Exclude test keys from the dataset
    remaining_df = target_df.drop(index=test_keys)

    calib_keys = None
    if calib_size is not None:
        # Calculate the number of calibration samples
        if isinstance(calib_size, float):
            n_calib = int(target_df.shape[0] * calib_size)
        elif isinstance(calib_size, int):
            n_calib = calib_size
        else:
            raise ValueError("calib_size must be either an int or a float.")

        # Get calibration keys
        calib_keys = get_stratified_sample_keys(remaining_df, stratify_target_col, size=n_calib,
                                                shuffle=shuffle, random_seed=random_seed, verbose=verbose)

        # Exclude calibration keys from the dataset
        remaining_df = remaining_df.drop(index=calib_keys)

    # Remaining keys are for the train set
    train_keys = remaining_df.index

    if verbose:
        print(
            f"Split completed: {len(train_keys)} train, {len(calib_keys) if calib_keys is not None else 0} calibration, {len(test_keys)} test samples.")

    if verbose:
        n_samples = target_df.shape[0]
        msg = (f"Stratified Train/Calib/Test Split. Data is split into:\n"
               f"    – train: {len(train_keys)} samples ({len(train_keys) / n_samples:.2%})\n")
        if calib_size is not None:
            msg = f"{msg}\n    – calibration: {len(calib_keys)} samples ({len(calib_keys) / n_samples:.2%})\n"
        msg = f"{msg}\n    – test: {len(test_keys)} samples ({len(test_keys) / n_samples:.2%})"
        logger.info(msg)
    return train_keys, calib_keys, test_keys


def get_stratified_train_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str,
                                         test_size: Union[int, float],
                                         shuffle: bool = True, random_seed: int = 0,
                                         verbose: bool = True) -> Tuple[pd.Index, pd.Index]:
    train_keys, _, test_keys = get_stratified_train_calib_test_split_keys(target_df, stratify_target_col,
                                                                          test_size=test_size,
                                                                          shuffle=shuffle,
                                                                          random_seed=random_seed,
                                                                          verbose=verbose)
    return train_keys, test_keys
