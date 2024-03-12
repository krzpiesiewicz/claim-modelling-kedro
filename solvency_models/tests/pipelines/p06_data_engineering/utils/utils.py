from typing import List, Dict, Tuple

import pandas as pd


def cat_col(lst: List) -> pd.Series:
    return ("cat_" + pd.Series(lst, dtype=str)).to_list()


def concat(lists: List[List]) -> List:
    return [item for sublist in lists for item in sublist]


def dicts_of_test_cases_to_input_and_output_dataframes(test_cases_data: Dict[str, Dict[str, List]]) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    input_data = {test_case: pd.Series(test_data["input"]) for test_case, test_data in test_cases_data.items()}
    output_data = {test_case: pd.Series(test_data["output"]) for test_case, test_data in test_cases_data.items()}
    return pd.DataFrame(input_data), pd.DataFrame(output_data)
