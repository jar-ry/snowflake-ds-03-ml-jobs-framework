from snowflake.snowpark import DataFrame


def load_data(customer_data: DataFrame, behavior_data: DataFrame) -> DataFrame:
    """Join CUSTOMERS and PURCHASE_BEHAVIOR tables on CUSTOMER_ID.

    Performs a left join and selects the columns needed for downstream
    feature engineering and modelling.
    """
    raw_data = customer_data.join(
        behavior_data,
        (customer_data["CUSTOMER_ID"] == behavior_data["CUSTOMER_ID"]),
        "left",
    ).rename(
        {
            customer_data["UPDATED_AT"]: "CUSTOMER_UPDATED_AT",
            customer_data["CUSTOMER_ID"]: "CUSTOMER_ID",
            behavior_data["UPDATED_AT"]: "BEHAVIOR_UPDATED_AT",
        }
    )

    return raw_data[
        [
            "CUSTOMER_ID",
            "AGE",
            "GENDER",
            "STATE",
            "ANNUAL_INCOME",
            "LOYALTY_TIER",
            "TENURE_MONTHS",
            "SIGNUP_DATE",
            "CUSTOMER_UPDATED_AT",
            "AVG_ORDER_VALUE",
            "PURCHASE_FREQUENCY",
            "RETURN_RATE",
            "LIFETIME_VALUE",
            "LAST_PURCHASE_DATE",
            "TOTAL_ORDERS",
            "BEHAVIOR_UPDATED_AT",
        ]
    ]
