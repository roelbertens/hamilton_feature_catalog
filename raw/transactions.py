import pyspark.sql as ps
from hamilton.function_modifiers import schema, tag

TRANSACTIONS__SCHEMA = [
    ("TRANSACTION_ID", "int"),
    ("TX_DATETIME", "timestamp"),
    ("CUSTOMER_ID", "int"),
    ("TERMINAL_ID", "int"),
    ("TX_AMOUNT", "double"),
    ("TX_TIME_SECONDS", "int"),
    ("TX_TIME_DAYS", "int"),
]


@tag(data_type="raw_data")
@schema.output(*TRANSACTIONS__SCHEMA)
def transactions(spark_session: ps.SparkSession) -> ps.DataFrame:
    # NOTE: to simulate we area dealing with big data we are going to use spark
    return spark_session.read.csv(
        "data/Final Transactions.csv", header=True, inferSchema=True
    ).select(*[i[0] for i in TRANSACTIONS__SCHEMA])
