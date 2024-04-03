import pyspark.sql as ps
from hamilton.function_modifiers import schema, tag
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from hamilton_feature_catalog.utils import spark_schema_to_col_name_type_tuples

TRANSACTION_FEATURES__SCHEMA = StructType(
    [
        StructField("sum_tx_amount", DoubleType(), True),
        StructField("mean_tx_amount", DoubleType(), True),
        StructField("median_tx_amount", DoubleType(), True),
    ]
)
TRANSACTION_FEATURES__SUPPORTED_LEVELS = ["customer_id"]


@tag(data_type="features")
@schema.output(*spark_schema_to_col_name_type_tuples(TRANSACTION_FEATURES__SCHEMA))
def transaction_features(
    transactions_clean: ps.DataFrame,
    number_of_decimals: int,
    aggregation_level: str,
) -> ps.DataFrame:
    """Features that describe the transaction behavior of a customer.

    Args:
        transactions_clean: one row per transaction
        number_of_decimals: number of decimals to round the features to
        aggregation_level: the level at which to aggregate the features

    Returns:
        sum_tx_amount: sum of all transactions
        mean_tx_amount: mean of all transactions
        median_tx_amount: median of all transactions
    """
    return (
        transactions_clean.groupby(aggregation_level)
        .agg(
            sf.round(sf.sum("tx_amount"), number_of_decimals).alias("sum_tx_amount"),
            sf.round(sf.mean("tx_amount"), number_of_decimals).alias("mean_tx_amount"),
            sf.round(sf.median("tx_amount"), number_of_decimals).alias(
                "median_tx_amount"
            ),
        )
        .select(aggregation_level, *TRANSACTION_FEATURES__SCHEMA.fieldNames())
    )
