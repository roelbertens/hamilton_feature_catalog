from typing import Optional

import pyspark.sql as ps
from hamilton.function_modifiers import (
    ResolveAt,
    pipe,
    resolve,
    schema,
    source,
    step,
    tag_outputs,
)
from pyspark.sql import functions as sf

TRANSACTIONS_CLEAN__SCHEMA = [
    ("transaction_id", "int"),
    ("tx_datetime", "timestamp"),
    ("customer_id", "int"),
    ("terminal_id", "int"),
    ("tx_amount", "double"),
    ("tx_time_seconds", "int"),
    ("tx_time_days", "int"),
]


def _is_special_customer(
    transactions: ps.DataFrame, special_customers: ps.DataFrame
) -> ps.DataFrame:
    return transactions.join(
        special_customers.withColumn("is_special_customer", sf.lit(True)),
        on="customer_id",
        how="left",
    ).fillna(False)


@tag_outputs(
    transactions_clean={"data_type": "intermediate"},
    with__is_special_customer={"data_type": "optional"},
)
@resolve(
    when=ResolveAt.CONFIG_AVAILABLE,
    decorate_with=lambda transactions_clean__with__is_special_customer: schema.output(
        *TRANSACTIONS_CLEAN__SCHEMA
        + (
            [("is_special_customer", "boolean")]
            if transactions_clean__with__is_special_customer
            else []
        ),
        target_="transactions_clean",
    ),
)
@pipe(
    step(_is_special_customer, special_customers=source("special_customers"))
    .named("with__is_special_customer", namespace=None)
    .when(transactions_clean__with__is_special_customer=True),
)
def transactions_clean(
    transactions: ps.DataFrame,
    transactions_clean__with__is_special_customer: bool,
    transactions__filter_expression: Optional[ps.column.Column] = None,
) -> ps.DataFrame:
    final_columns = [i[0] for i in TRANSACTIONS_CLEAN__SCHEMA]
    if transactions_clean__with__is_special_customer:
        final_columns.append("is_special_customer")

    if transactions__filter_expression is not None:
        transactions = transactions.filter(transactions__filter_expression)

    return transactions.distinct().dropna().select(*final_columns)
