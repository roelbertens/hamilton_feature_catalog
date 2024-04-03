from datetime import date

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from hamilton_feature_catalog.utils import (
    IncorrectScope,
    check_exactly_one_is_none,
    check_scope,
    check_type,
    get_dates,
)


def test_get_dates():
    dates = get_dates(date(2023, 1, 5), date(2023, 1, 6), 2)
    assert isinstance(dates, tuple), "Result should be of type 'tuple'"
    assert dates == ("2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08")


def test_check_type_decorator():
    @check_type
    def multiply(x: int, y: int) -> int:
        return x * y

    try:
        multiply(3, 4)  # This should pass as both arguments are of type int
        multiply(
            3, "5"
        )  # This should raise a TypeError as the second argument is not of type int
    except TypeError as e:
        assert str(e) == "Argument '5' is not of type 'int'"
    else:
        assert False, "Expected TypeError was not raised"


def test_check_scope(spark: SparkSession):
    aggregation_level = "id"
    scope = spark.createDataFrame([["1"]], schema=[aggregation_level])
    check_scope(
        scope=scope,
        aggregation_level=aggregation_level,
    )


def test_check_scope__wrong_column(spark: SparkSession):
    aggregation_level = "id"
    scope = spark.createDataFrame([["1"]], schema=["wrong_column"])
    with pytest.raises(IncorrectScope):
        check_scope(
            scope=scope,
            aggregation_level=aggregation_level,
        )


def test_check_scope__too_many_columns(spark: SparkSession):
    aggregation_level = "id"
    scope = spark.createDataFrame([["1", "b"]], schema=["id", "additional_column"])
    with pytest.raises(IncorrectScope):
        check_scope(
            scope=scope,
            aggregation_level=aggregation_level,
        )


def test_check_scope__wrong_type(spark: SparkSession):
    aggregation_level = "id"
    scope = pd.DataFrame([["1"]], columns=["id"])
    with pytest.raises(IncorrectScope):
        check_scope(
            scope=scope,
            aggregation_level=aggregation_level,
        )


def test_check_exactly_one_is_none(spark: SparkSession):
    filter_expression = F.lit(True)
    example_date = date(2022, 1, 1)
    assert check_exactly_one_is_none(None, example_date)
    assert check_exactly_one_is_none(example_date, None)
    assert check_exactly_one_is_none(None, filter_expression)
    assert check_exactly_one_is_none(filter_expression, None)
    assert check_exactly_one_is_none(filter_expression, example_date) == False
    assert check_exactly_one_is_none(example_date, filter_expression) == False
