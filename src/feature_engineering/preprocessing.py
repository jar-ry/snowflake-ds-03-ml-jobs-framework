import snowflake.snowpark.functions as F
from snowflake.snowpark import DataFrame


def pre_process(data: DataFrame) -> DataFrame:
    """Derive features from raw customer/behavior data.

    Computes AVERAGE_ORDER_PER_MONTH, DAYS_SINCE_LAST_PURCHASE, DAYS_SINCE_SIGNUP,
    EXPECTED_DAYS_BETWEEN_PURCHASES, and DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE.
    Division-by-zero cases (zero tenure or purchase frequency) return NULL.
    """
    data = data.with_column("ANNUAL_INCOME", F.round(F.col("ANNUAL_INCOME"), 0))

    days_since_last = F.datediff("day", F.col("LAST_PURCHASE_DATE"), F.col("BEHAVIOR_UPDATED_AT"))
    expected_days = F.iff(
        F.col("PURCHASE_FREQUENCY") == 0,
        F.lit(None),
        F.lit(30) / F.col("PURCHASE_FREQUENCY"),
    )

    data = data.with_columns(
        [
            "AVERAGE_ORDER_PER_MONTH",
            "DAYS_SINCE_LAST_PURCHASE",
            "DAYS_SINCE_SIGNUP",
            "EXPECTED_DAYS_BETWEEN_PURCHASES",
            "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
        ],
        [
            F.iff(
                F.col("TENURE_MONTHS") == 0,
                F.lit(None),
                F.col("TOTAL_ORDERS") / F.col("TENURE_MONTHS"),
            ),
            days_since_last,
            F.datediff("day", F.col("SIGNUP_DATE"), F.col("BEHAVIOR_UPDATED_AT")),
            expected_days,
            F.round(days_since_last - expected_days, 0),
        ],
    )

    return data
