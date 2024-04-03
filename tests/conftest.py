import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="session")
def aggregation_level() -> str:
    return "customer_id"


@pytest.fixture(scope="session")
def scope(spark: SparkSession, aggregation_level: str) -> str:
    schema = StructType([StructField(aggregation_level, StringType(), True)])
    return spark.createDataFrame([], schema=schema)


@pytest.fixture(scope="session")
def number_of_decimals() -> int:
    return 1
