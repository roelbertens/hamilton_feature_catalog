import pyspark.sql as ps
from hamilton.function_modifiers import schema, tag
from pyspark.sql import functions as sf

TRANSACTION_FEATURES__SCHEMA = [
    ("customer_id", "int"),
    ("sum_tx_amount", "double"),
    ("mean_tx_amount", "double"),
    ("median_tx_amount", "double"),
]


@tag(data_type="features")
@schema.output(*TRANSACTION_FEATURES__SCHEMA)
def transaction_features(
    transactions_clean: ps.DataFrame,
    number_of_decimals: int,
) -> ps.DataFrame:

    return transactions_clean.groupby("customer_id").agg(
        sf.round(sf.sum("tx_amount"), number_of_decimals).alias("sum_tx_amount"),
        sf.round(sf.mean("tx_amount"), number_of_decimals).alias("mean_tx_amount"),
        sf.round(sf.median("tx_amount"), number_of_decimals).alias("median_tx_amount"),
    )
