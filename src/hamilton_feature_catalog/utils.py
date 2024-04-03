import importlib
import logging
import os
import re
import sys
from datetime import date, timedelta
from functools import wraps
from typing import Any, Callable

import IPython
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType


def get_dates(
    start_date: date, end_date: date, additional_days_at_end: int
) -> tuple[str, ...]:
    """Get a tuple of dates (str format) from start to end date, extended with the specified additional days at the end

    Example:
    > get_dates(date(2023,1, 5), date(2023,1, 6), 1)
    > ("2023-01-05", "2023-01-06", "2023-01-07")

    Args:
        start_date: first date
        end_date: last date
        additional_days_at_end: number of additional days after the last date.

    Returns:
        A tuple containing a string for each date.
    """
    return tuple(
        (start_date + timedelta(days=x)).strftime("%Y-%m-%d")
        for x in range((end_date - start_date).days + 1 + additional_days_at_end)
    )


def check_type(func: Callable) -> Callable:
    """Check if all function arguments are of the specified type"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        for arg, expected_type in zip(args, func.__annotations__.values()):
            if not isinstance(arg, expected_type):
                raise TypeError(
                    f"Argument '{arg}' is not of type '{expected_type.__name__}'"
                )
        return func(*args, **kwargs)

    return wrapper


def create_feature_overview() -> pd.DataFrame:
    """Create overview with all available features, incl.: name, description and module"""
    features_per_module = []

    for module in import_and_list_modules("features"):
        if module.__name__ == "__init__":
            continue
        docstring = getattr(module, module.__name__).__doc__
        names_and_descriptions = extract_feature_names_and_descriptions_from_docstring(
            docstring
        )
        descriptions = pd.DataFrame(
            [
                names_and_descriptions[i : i + 2]
                for i in range(0, len(names_and_descriptions), 2)
            ],
            columns=["name", "description"],
        )

        schema = getattr(module, f"{module.__name__.upper()}__SCHEMA")
        types = pd.DataFrame(
            [
                {"name": field.name, "type": str(field.dataType)}
                for field in schema.fields
            ],
            columns=["name", "type"],
        )

        supported_levels = getattr(
            module, f"{module.__name__.upper()}__SUPPORTED_LEVELS"
        )

        overview = descriptions.merge(types, on="name", how="outer")
        overview["aggregation levels"] = str(supported_levels)
        overview["module"] = module.__name__

        features_per_module.append(overview)

    return pd.concat(features_per_module).reset_index(drop=True)


def get_supported_feature_names() -> list[str]:
    """Return the names of all available features"""
    feature_overview = create_feature_overview()
    return feature_overview["name"].unique().tolist()


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s - %(message)s",
    )
    return logging.getLogger()


def spark_schema_to_col_name_type_tuples(schema: StructType) -> list[tuple[str, str]]:
    """Extract the column name and type from a StructType and put them in a list of tuples.

    E.g.
    ```
    schema = StructType([StructField("col_a", StringType(), True), StructField("col_b", IntegerType(), True)])
    spark_schema_to_col_name_type_tuples(schema)
    > [("col_a", "string"), ("col_b", "integer")]
    ```

    Used when defining the schema.output for Hamilton nodes

    NOTE: also removes the 'Type()' part and makes it lowercase.
    """

    def _remove_type(string: str) -> str:
        return re.sub("Type$", "", string)

    def _remove_special_chars(string: str) -> str:
        return re.match("^[a-zA-Z]*", string).group()

    def clean(string: str) -> str:
        return _remove_type(_remove_special_chars(string)).lower()

    return [(field.name, clean(str(field.dataType))) for field in schema.fields]


def check_if_aggregation_level_is_supported(
    aggregation_level: str, supported_levels: list[str], logger: logging.Logger
) -> None:
    """Verify if the specified aggregation_level is part of the supported_levels"""
    if aggregation_level not in supported_levels:
        error_message = f"Error: aggregation level '{aggregation_level}' not in supported levels: {supported_levels}."
        logger.error(error_message)
        raise UnsupportedAggregationLevel(error_message)


def extract_feature_names_and_descriptions_from_docstring(docstring: str) -> list[str]:
    """To allow for automatic documentation we assume the 'Returns:' section to specify
    EXACTLY one line per feature in format 'name: description'
    and leaving out the 'aggregation_level' column
    """
    split_string = "Returns:"
    if split_string not in docstring:
        raise IncorrectDocstring(
            "Expected '{split_string}' in docstring to identify feature names and descriptions."
        )
    docstring_returns_section = docstring.split(split_string)[-1].strip()

    return [item.strip() for item in re.split(":|\n", docstring_returns_section)]


def get_feature_names_from_feature_groups_docstring(docstring: str) -> list[str]:
    """Get the list of available features from the docstring"""
    return extract_feature_names_and_descriptions_from_docstring(docstring)[0::2]


def import_and_list_modules(directory: str) -> list[Any]:
    """Imports and returns all modules from the specified directory

    Args:
        directory: relative from the root_directory
    """
    root_directory = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(root_directory, directory)
    sys.path.append(directory)

    modules = []
    for _, dirs, files in os.walk(directory):
        for dir in dirs:
            sys.path.append(os.path.join(directory, dir))
        for filename in files:
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                module = importlib.import_module(module_name)
                modules.append(module)

    sys.path.remove(directory)
    return modules


def check_scope(
    scope: SparkDataFrame, aggregation_level: str, logger: logging.Logger = get_logger()
) -> None:
    """Verify if the scope is of the expected format given the aggregation_level"""
    message = f"Error: scope should be spark df with exactly one column named '{aggregation_level}', but got {type(scope)} with columns: {scope.columns}"
    if type(scope) != SparkDataFrame:
        logger.error(message)
        raise IncorrectScope(message)
    if len(scope.columns) != 1:
        logger.error(message)
        raise IncorrectScope(message)
    if scope.columns[0] != aggregation_level:
        logger.error(message)
        raise IncorrectScope(message)


def check_exactly_one_is_none(a: Any, b: Any) -> bool:
    """Return True when exactly one of a and b is None"""
    return (a is not None) ^ (b is not None)


def get_dbutils(spark: SparkSession) -> Any:
    dbutils = None
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
    else:
        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils


class UnsupportedAggregationLevel(Exception):
    pass


class UnsupportedFeatureName(Exception):
    pass


class IncorrectScope(Exception):
    pass


class MissingInput(Exception):
    pass


class IncompatibleInput(Exception):
    pass


class IncorrectDocstring(Exception):
    pass
