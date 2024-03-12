from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest

from solvency_models.pipelines.p06_data_engineering.utils.cat_reducer import fit_transform_categories_reducer
from .utils import dicts_of_test_cases_to_input_and_output_dataframes, concat, cat_col


@dataclass
class DEConfigReduceCategories:
    reducer_max_categories: int
    reducer_min_frequency: int
    reducer_join_infrequent: str
    reducer_join_exceeding_max_categories: str
    is_cat_reducer_enabled: bool = True
    join_exceeding_and_join_infrequent_values_warning = "Both parameters join_exceeding_max_categories and join_infrequent are provided"

    def __init__(self, reducer_max_categories: int = None, reducer_min_frequency: int = None,
                 reducer_join_infrequent: str = None,
                 reducer_join_exceeding_max_categories: str = None):
        self.reducer_max_categories = reducer_max_categories
        self.reducer_min_frequency = reducer_min_frequency
        self.reducer_join_infrequent = reducer_join_infrequent
        self.reducer_join_exceeding_max_categories = reducer_join_exceeding_max_categories


@dataclass
class MockReduceCategoriesConfig:
    de: DEConfigReduceCategories

    def __init__(self, de: DEConfigReduceCategories):
        self.de = de


@pytest.fixture
def mock_reduce_categories_config(request):
    return MockReduceCategoriesConfig(de=request.param)


@pytest.fixture
def mock_reduce_categories_data(request):
    return dicts_of_test_cases_to_input_and_output_dataframes(request.param)


@patch("mlflow.log_artifact")
def _test_reduce_categories(mock_log_artifact, mock_reduce_categories_config, mock_reduce_categories_data):
    mock_log_artifact.return_value = None  # Set the return value of the mocked function
    mock_reduce_categories_input_data, mock_reduce_categories_output_data = mock_reduce_categories_data
    result = fit_transform_categories_reducer(mock_reduce_categories_config, mock_reduce_categories_input_data)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result.sort_index(axis=1), mock_reduce_categories_output_data.sort_index(axis=1))
    mock_log_artifact.assert_called_once()  # Check that the mocked function was called exactly once


@pytest.mark.parametrize("mock_reduce_categories_config", [
    DEConfigReduceCategories(reducer_max_categories=3, reducer_min_frequency=2,
                             reducer_join_exceeding_max_categories="other"),
], indirect=True)
@pytest.mark.parametrize("mock_reduce_categories_data", [
    dict(
        test_case_1=dict(
            input=cat_col(concat([[1] * 2, [2] * 2, [3] * 2, [4] * 2, [5] * 2])),
            output=cat_col(concat([[1] * 2, [2] * 2, [3] * 2])) + ["other"] * 4,
        ),
        test_case_2=dict(
            input=cat_col(concat([[1] * 2, [2, 3, 4, 5, 6, 7, 8, 9]])),
            output=cat_col(concat([[1] * 2])) + ["other"] * 8,
        )
    )], indirect=True)
def test_reduce_categories_join_other(mock_reduce_categories_config, mock_reduce_categories_data):
    _test_reduce_categories(mock_reduce_categories_config=mock_reduce_categories_config,
                            mock_reduce_categories_data=mock_reduce_categories_data)


@pytest.mark.parametrize("mock_reduce_categories_config", [
    DEConfigReduceCategories(reducer_min_frequency=2,
                             reducer_join_infrequent="infrequent"),
], indirect=True)
@pytest.mark.parametrize("mock_reduce_categories_data", [
    dict(
        test_case_1=dict(
            input=cat_col(concat([[1] * 3, [2] * 2, [3, 4, 5, 6, 7]])),
            output=cat_col(concat([[1] * 3, [2] * 2])) + ["infrequent"] * 5,
        ),
        test_case_2=dict(
            input=cat_col(concat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])),
            output=["infrequent"] * 10,
        ),
        test_case_3=dict(
            input=cat_col(concat([[i] * 2 for i in range(1, 6)])),
            output=cat_col(concat([[i] * 2 for i in range(1, 6)])),
        )
    )], indirect=True)
def test_reduce_categories_join_infrequent(mock_reduce_categories_config, mock_reduce_categories_data):
    _test_reduce_categories(mock_reduce_categories_config=mock_reduce_categories_config,
                            mock_reduce_categories_data=mock_reduce_categories_data)


@pytest.mark.parametrize("mock_reduce_categories_config", [
    DEConfigReduceCategories(reducer_max_categories=5, reducer_min_frequency=2,
                             reducer_join_exceeding_max_categories="other",
                             reducer_join_infrequent="infrequent"),
], indirect=True)
@pytest.mark.parametrize("mock_reduce_categories_data", [
    dict(
        test_case_1=dict(
            input=cat_col(concat([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])),
            output=["other"] * 10,
        ),
    )], indirect=True)
def test_reduce_categories_join_both(mock_reduce_categories_config, mock_reduce_categories_data):
    _test_reduce_categories(mock_reduce_categories_config=mock_reduce_categories_config,
                            mock_reduce_categories_data=mock_reduce_categories_data)


@pytest.mark.parametrize("mock_reduce_categories_config", [
    DEConfigReduceCategories(),
], indirect=True)
@pytest.mark.parametrize("mock_reduce_categories_data", [
    dict(
        test_case_1=dict(
            input=["C'mon, Ćma \"Dziwna\" chodziła nocą po łące    'ł.ą.c.k.a' !!! O Matko."],
            output=["Cmon_Cma_Dziwna_chodzila_noca_po_lace_lacka_O_Matko"],
        ),
    )], indirect=True)
def test_reduce_categories_escape_names(mock_reduce_categories_config, mock_reduce_categories_data):
    _test_reduce_categories(mock_reduce_categories_config=mock_reduce_categories_config,
                            mock_reduce_categories_data=mock_reduce_categories_data)
