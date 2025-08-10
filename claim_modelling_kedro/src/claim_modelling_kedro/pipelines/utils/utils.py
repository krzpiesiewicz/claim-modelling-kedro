import importlib
import logging
import math
import os
import shutil
from typing import Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def parse_catalog_as_dict(catalog_path=None):
    """
    Function to parse the Kedro catalog.yml as a dictionary.

    Args:
        catalog_path (str): The path to the catalog.yml file.
                            If None, defaults to 'conf/base/catalog.yml'.

    Returns:
        dict: Parsed catalog.yml content as a dictionary.
    """
    if catalog_path is None:
        catalog_path = os.path.join("conf", "base", "catalog.yml")

    # Check if the catalog.yml file exists
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog file not found at {catalog_path}")

    # Load the catalog.yml file
    with open(catalog_path, 'r') as file:
        catalog_dict = yaml.safe_load(file)

    return catalog_dict


def remove_file_or_directory(file_path):
    """
    Check if the file or directory exists, and remove it.

    Args:
        file_path (str): The path to the file or directory to be removed.
    """
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            # If it's a directory, remove the directory and all its contents
            shutil.rmtree(file_path)
            logger.info(f"Directory '{file_path}' removed.")
        else:
            # If it's a file, remove the file
            os.remove(file_path)
            logger.info(f"File '{file_path}' removed.")
    else:
        logger.info(f"File or directory '{file_path}' does not exist.")


def get_class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_constructor = getattr(module, class_name)
    return class_constructor


def shortened_seq_to_string(seq: Sequence) -> str:
    def val_to_str(val):
        return f"{val:.3f}"

    lst = [val_to_str(elem) for elem in seq]
    if len(lst) <= 10:
        return ", ".join(lst)
    else:
        return ", ".join(lst[:5]) + " ... " + ", ".join(lst[-5:])


def convert_np_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(i) for i in obj]
    else:
        return obj


def round_decimal(val: float, significant_digits: int) -> float | int:
    if val is None:
        return None
    if not np.isfinite(val):
        return val
    if val == 0:
        return 0

    abs_val = abs(val)
    int_part = math.floor(abs_val)
    num_digits_int_part = len(str(int_part)) if int_part > 0 else 0

    # If all significant digits are taken by the integer part → return original value
    if num_digits_int_part >= significant_digits:
        return int(val) if val == int(val) else math.copysign(int_part, val)

    # Digits left for decimal part
    decimal_digits = significant_digits - num_digits_int_part

    rounded = round(val, decimal_digits)
    return rounded
