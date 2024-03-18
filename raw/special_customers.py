import pyspark.sql as ps
from hamilton.function_modifiers import schema, tag

SPECIAL_CUSTOMERS__SCHEMA = [("customer_id", "int")]


@tag(data_type="raw_data")
@schema.output(*SPECIAL_CUSTOMERS__SCHEMA)
def special_customers(spark_session: ps.SparkSession) -> ps.DataFrame:
    return (
        spark_session.read.csv(
            "data/special_customers.csv", header=True, inferSchema=True
        )
        .select(*[i[0] for i in SPECIAL_CUSTOMERS__SCHEMA])
        .distinct()
    )
