import importlib
from typing import Sequence, Union

import numpy as np
import pandas as pd


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


def assert_pandas_no_lacking_indexes(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     target_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     source_df_name: str,
                                     target_df_name: str):
    if type(source_df) is pd.DataFrame or type(source_df) is pd.Series:
        if not source_df.index.intersection(target_df.index).equals(source_df.index):
            raise ValueError(f"""{target_df_name} does not have some indexes from {source_df_name}:
    - {source_df_name}.index: {source_df.index}
    - {target_df_name}.index: {target_df.index}
    - lacking indexes: {source_df.index.difference(target_df.index)}""")


def trunc_target_index(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                       target_df: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    if type(source_df) is pd.DataFrame:
        target_df = target_df.loc[source_df.index, :]
    if type(source_df) is pd.Series:
        target_df = target_df.loc[source_df.index]
    return target_df


def preds_as_dataframe_with_col_name(source_df: Union[pd.DataFrame, pd.Series, np.ndarray],
                                     preds: Union[pd.DataFrame, pd.Series, np.ndarray], pred_col: str) -> pd.DataFrame:
    if type(preds) is np.ndarray:
        if len(preds.shape) == 1:
            preds = pd.Series(preds)
        else:
            if preds.shape[0] != source_df.shape[0]:
                preds = preds.transpose()
            preds = pd.DataFrame(preds)
    if type(preds) is pd.Series:
        preds = pd.DataFrame(preds)
    if len(preds.columns) == 1:
        preds.columns = [pred_col]
    if not preds.index.equals(source_df.index):
        preds.set_index(source_df.index, inplace=True)
    return preds
