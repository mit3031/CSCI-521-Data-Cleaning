import pandas as pd


"""
Fills missing emails and marks the placeholder values
"""
def handle_missing_emails(customers_df):
    missing_count = customers_df["email"].isna().sum()
    customers_df["email"] = customers_df["email"].fillna("unknown@missing.local")

    # Also marks the unknown customer added by Role A
    customers_df["email_is_placeholder"] = customers_df["email"].eq("unknown@missing.local")
    print(f"Filled {missing_count} missing emails")
    return customers_df


"""
Fills missing order dates with the mode and flags inverted dates
"""
def handle_date_logic(orders_df):
    order_dates = pd.to_datetime(orders_df["order_date"], format="mixed", errors="coerce")
    ship_dates = pd.to_datetime(orders_df["ship_date"], format="mixed", errors="coerce")
    missing_dates = orders_df["order_date"].isna()
    mode_dates = order_dates.mode()

    # If there is a tie, mode() gives us the earliest date first
    if not mode_dates.empty:
        mode_date = mode_dates.iloc[0]
        orders_df.loc[missing_dates, "order_date"] = mode_date.strftime("%Y-%m-%d")
        order_dates.loc[missing_dates] = mode_date
        print(f"Filled {missing_dates.sum()} missing order dates with {mode_date:%Y-%m-%d}")
    else:
        print("No valid order dates available for mode imputation")

    # Check after filling missing dates, but don't swap or change inverted dates
    orders_df["ship_before_order_flag"] = ship_dates < order_dates
    print(f"Flagged {orders_df['ship_before_order_flag'].sum()} orders with ship dates before order dates")
    return orders_df


"""
Converts the date columns to YYYY-MM-DD
"""
def standardize_all_dates(df, date_cols):
    for col in date_cols:
        parsed_dates = pd.to_datetime(df[col], format="mixed", errors="coerce")
        df[col] = parsed_dates.dt.strftime("%Y-%m-%d")
    return df
