import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _interval_midpoint(interval):
    """Helper function to calculate the midpoint of a pandas.Interval."""
    return (interval.left + interval.right) / 2


def _merge_infrequent_bins(stratified_bins, min_samples=2):
    """ Merge bins with fewer than min_samples into adjacent bins based on midpoints. """
    bin_counts = pd.Series(stratified_bins).value_counts()

    # Identify infrequent bins (those with less than min_samples)
    infrequent_bins = bin_counts[bin_counts < min_samples].index
    frequent_bins = bin_counts[bin_counts >= min_samples].index

    # Convert stratified_bins to DataFrame for easier manipulation
    stratified_df = pd.DataFrame({'bins': stratified_bins})

    # Get midpoints of all bins
    bin_midpoints = pd.Series({bin: _interval_midpoint(bin) for bin in infrequent_bins.union(frequent_bins)})

    for inf_bin in infrequent_bins:
        # Get the midpoint of the infrequent bin
        inf_bin_midpoint = bin_midpoints[inf_bin]

        # Find the closest frequent bin by comparing midpoints
        closest_bin = (bin_midpoints[frequent_bins] - inf_bin_midpoint).abs().idxmin()

        # Merge the infrequent bin with the closest frequent bin
        stratified_df['bins'].replace(inf_bin, closest_bin, inplace=True)

    return stratified_df['bins']


def _get_stratified_train_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str, test_size: float,
                                          shuffle: bool = True, random_seed: int = 0, verbose: bool = True,
                                          split_type: str = 'train_test', calib_size: float = None):
    """
    Splits the dataset into stratified train/test or train/calib/test sets using either a simple split or three-way split.

    Args:
        target_df (pd.DataFrame): The DataFrame containing the target column.
        stratify_target_col (str): The column in target_df to stratify on (e.g., a continuous variable like frequency/severity).
        test_size (float): The test set size (as a fraction between 0 and 1).
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        random_seed (int): The random seed for reproducibility. Default is 0.
        verbose (bool): Whether to print information about the splits. Default is True.
        split_type (str): The type of split. Either 'train_test' for a standard train/test split or 'train_calib_test' for a train/calib/test split.
        calib_size (float, optional): The calibration set size (as a fraction between 0 and 1). If not provided, it will be set equal to test_size.

    Returns:
        Tuple[pd.Index, pd.Index, pd.Index]: Indices of the training set, calibration set (optional), and test set.
    """

    # Ensure calib_size is provided or defaults to test_size if using 'train_calib_test'
    if split_type == 'train_calib_test' and calib_size is None:
        calib_size = test_size

    # Total number of samples
    n_samples = target_df.shape[0]

    # Calculate the minimum number of samples per bin based on the test size or train size, whichever is smaller
    min_ratio = np.min([test_size, calib_size, 1 - test_size - calib_size] if split_type == 'train_calib_test'
                       else [test_size, 1 - test_size])
    min_samples_per_bin = np.ceil(1 / min_ratio)

    # Calculate the number of bins to ensure at least `min_samples_per_bin` samples per bin
    n_bins = max(2, int(np.floor(n_samples / min_samples_per_bin)))

    # Ensure the number of bins does not exceed a reasonable limit
    max_bins = int(n_samples * test_size / min_samples_per_bin)
    n_bins = min(n_bins, max_bins)

    # Bin the continuous variable using pandas.qcut to create quantile-based bins for fair stratification
    stratified_bins = pd.qcut(target_df[stratify_target_col], q=n_bins, duplicates='drop')

    # Merge infrequent bins into adjacent bins
    stratified_bins = _merge_infrequent_bins(stratified_bins, min_samples=2)

    stratified_labels, _ = pd.factorize(stratified_bins)

    if split_type == 'train_test':
        # Perform a simple train/test split
        train_policies, test_policies = train_test_split(
            target_df,
            test_size=test_size,
            stratify=stratified_labels,
            random_state=random_seed,
            shuffle=shuffle
        )
        train_keys = train_policies.index
        test_keys = test_policies.index
        if verbose:
            logger.info(f"Stratified Train/Test Split. Data is split into:\n"
                        f"    - train: {len(train_keys)} samples ({len(train_keys) / n_samples:.2%})\n"
                        f"    - test: {len(test_keys)} samples  ({len(test_keys) / n_samples:.2%})")
        return train_keys, None, test_keys  # Calibration set is None for 'train_test'

    elif split_type == 'train_calib_test':
        # Step 1: Perform a train/test split with the given test_size
        train_policies, test_policies = train_test_split(
            target_df,
            test_size=test_size,
            stratify=stratified_labels,
            random_state=random_seed,
            shuffle=shuffle
        )

        # Calculate calibration size relative to the remaining training set
        test_size_actual = len(test_policies) / n_samples  # Test set size as a fraction
        calib_size_relative = calib_size / (
                1 - test_size_actual)  # Adjust calib_size relative to remaining training data

        # Step 2: Split remaining train_policies into train and calibration sets

        # Bin the continuous variable using pandas.qcut to create quantile-based bins for fair stratification
        stratified_bins_train = pd.qcut(train_policies[stratify_target_col], q=n_bins, duplicates='drop')

        # Merge infrequent bins into adjacent bins
        stratified_bins_train = _merge_infrequent_bins(stratified_bins_train, min_samples=2)

        stratified_labels_train, _ = pd.factorize(stratified_bins_train)

        train_policies_final, calib_policies = train_test_split(
            train_policies,
            test_size=calib_size_relative,  # Adjusted calib_size
            stratify=stratified_labels_train,
            random_state=random_seed,
            shuffle=shuffle
        )

        train_keys = train_policies_final.index
        calib_keys = calib_policies.index
        test_keys = test_policies.index

        if verbose:
            logger.info(f"Stratified Train/Calib/Test Split. Data is split into:\n"
                        f"    – train: {len(train_keys)} samples ({len(train_keys) / n_samples:.2%})\n"
                        f"    – calibration: {len(calib_keys)} samples ({len(calib_keys) / n_samples:.2%})\n"
                        f"    – test: {len(test_keys)} samples ({len(test_keys) / n_samples:.2%})")
        return train_keys, calib_keys, test_keys


def get_stratified_train_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str, test_size: float,
                                         shuffle: bool = True, random_seed: int = 0,
                                         verbose: bool = True) -> Tuple[pd.Index, pd.Index]:
    train_keys, _, test_keys = _get_stratified_train_test_split_keys(target_df, stratify_target_col,
                                                                     test_size=test_size,
                                                                     shuffle=shuffle,
                                                                     random_seed=random_seed,
                                                                     verbose=verbose,
                                                                     split_type='train_test')
    return train_keys, test_keys


def get_stratified_train_calib_test_split_keys(target_df: pd.DataFrame, stratify_target_col: str, test_size: float,
                                               calib_size: float,
                                               shuffle: bool = True, random_seed: int = 0, verbose: bool = True) -> \
        Tuple[pd.Index, pd.Index, pd.Index]:
    return _get_stratified_train_test_split_keys(target_df, stratify_target_col, test_size=test_size,
                                                 calib_size=calib_size, shuffle=shuffle, random_seed=random_seed,
                                                 verbose=verbose, split_type='train_calib_test')
