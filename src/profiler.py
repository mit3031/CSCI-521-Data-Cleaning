import pandas as pd
import os

# Paths
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(project_root, "output")

# output folder
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Added a basic state list for accurate state validation
US_STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
             "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
             "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
             "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
             "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]

"""
Data profile stats are below
"""
def profile_dataset(df, dataset_name):
    total_records = len(df)
    total_attributes = len(df.columns)

    stats_list = []

    for col in df.columns:
        col_type = df[col].dtype

        # count of missing
        missing_count = df[col].isnull().sum()

        # Only checks placeholders on rows that are not already null
        if col_type == "object":
            valid_strings = df[col].dropna().astype(str).str.strip().str.upper()
            bad_strings = ["", "NULL", "N/A", "NONE", "-"]
            missing_count += valid_strings.isin(bad_strings).sum()

        missing_percentage = (missing_count / total_records) * 100
        unique_count = df[col].nunique()

        # Mode
        mode_series = df[col].mode()
        col_mode = mode_series[0] if len(mode_series) > 0 else "N/A"

        # Default values for stats
        col_mean = col_median = col_std = "N/A"
        col_25 = col_50 = col_75 = "N/A"
        col_min = col_max = "N/A"

        # full summary stats for numerics
        if pd.api.types.is_numeric_dtype(df[col]):
            if "_id" not in col.lower():
                col_mean = round(df[col].mean(), 2)
                col_std = round(df[col].std(), 2)
                col_25 = round(df[col].quantile(0.25), 2)
                col_50 = round(df[col].quantile(0.50), 2)
                col_75 = round(df[col].quantile(0.75), 2)
                col_median = col_50
            col_min = round(df[col].min(), 2)
            col_max = round(df[col].max(), 2)

        # Parse to dates so min/max arent alphabetical
        elif "date" in col.lower():
            parsed_dates = pd.to_datetime(df[col], errors='coerce')
            valid_dates = parsed_dates.dropna()
            if len(valid_dates) > 0:
                col_min = valid_dates.min().date()
                col_max = valid_dates.max().date()
        else:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                col_min = valid_data.min()
                col_max = valid_data.max()

        stats_list.append({
            "Attribute": col,
            "Data Type": str(col_type),
            "Missing Count": int(missing_count),
            "Missing %": round(missing_percentage, 2),
            "Unique Values": unique_count,
            "Mode": col_mode,
            "Mean": col_mean,
            "Std": col_std,
            "Min": col_min,
            "25%": col_25,
            "Median (50%)": col_50,
            "75%": col_75,
            "Max": col_max
        })

    return pd.DataFrame(stats_list)

"""
Data Quality Evaluation checks below
"""

def check_duplicates(df, id_col, table_name, log_file):
    cols_to_check = [c for c in df.columns if c != id_col]
    dup_count = df.duplicated(subset=cols_to_check).sum()
    log_file.write(f"{table_name} exact duplicates (ignoring {id_col}): {dup_count}\n")
    return dup_count

def check_foreign_keys(parent_df, parent_col, child_df, child_col, relationship, log_file):
    valid_ids = parent_df[parent_col].dropna()
    orphans = child_df[~child_df[child_col].isin(valid_ids)]
    log_file.write(f"Integrity check ({relationship}) - Orphans found: {len(orphans)}\n")

def check_outliers_iqr(df, col_name, table_name, log_file):
    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col_name] < (q1 - 1.5 * iqr)) | (df[col_name] > (q3 + 1.5 * iqr))]
        log_file.write(f"{table_name} - {col_name} Statistical Outliers (IQR): {len(outliers)}\n")

def check_business_rules(customers, orders, items, products, log_file):
    log_file.write("\n--- Invalid Values & Format Checks ---\n")

    # Catches unparseable strings properly
    orders['dt_order'] = pd.to_datetime(orders['order_date'], errors='coerce')
    orders['dt_ship'] = pd.to_datetime(orders['ship_date'], errors='coerce')

    # If it wasnt null originally, but became NAT, it's an unparseable invalid format
    bad_order_dates = orders['dt_order'].isna() & orders['order_date'].notna()
    log_file.write(f"Orders with unparseable order_date formats: {bad_order_dates.sum()}\n")

    bad_dates = orders[orders['dt_ship'] < orders['dt_order']]
    log_file.write(f"Orders where ship date is logically before order date: {len(bad_dates)}\n")

    # Future Dates check
    today = pd.Timestamp.today()
    log_file.write(f"Orders from the future (order_date > today): {len(orders[orders['dt_order'] > today])}\n")

    # Text Format Checks
    emails = customers['email'].dropna().astype(str)
    bad_emails = emails[~emails.str.contains("@", na=False)]
    log_file.write(f"Customers with malformed emails (no '@'): {len(bad_emails)}\n")

    states = customers['state'].dropna().astype(str).str.strip().str.upper()
    bad_states = states[~states.isin(US_STATES)]
    log_file.write(f"Customers with invalid state codes: {len(bad_states)}\n")

    # Zip and Phone formatting checks
    zips = customers['zip_code'].dropna().astype(str).str.strip()
    bad_zips = zips[~zips.str.match(r'^\d{5}(-\d{4})?$')]
    log_file.write(f"Customers with malformed zip codes: {len(bad_zips)}\n")

    phones = customers['phone'].dropna().astype(str).str.replace(r'\D+', '', regex=True)
    bad_phones = phones[(phones.str.len() < 10) | (phones.str.len() > 11)]
    log_file.write(f"Customers with invalid phone numbers (digits != 10 or 11): {len(bad_phones)}\n")

    # Soft Duplicates Check
    email_dups = customers.duplicated(subset=['email'], keep=False).sum()
    log_file.write(f"Customers sharing the exact same email: {email_dups}\n")

    log_file.write("\n--- Outliers, Negatives, & Coercion Fails ---\n")

    # Catch strings hidden in numeric columns
    sales_num = pd.to_numeric(orders['sales_amount'], errors='coerce')
    bad_sales_format = sales_num.isna() & orders['sales_amount'].notna()
    log_file.write(f"Orders with unparseable sales_amount (e.g. text/symbols): {bad_sales_format.sum()}\n")

    discount_num = pd.to_numeric(orders['discount'], errors='coerce')
    tax_num = pd.to_numeric(orders['tax_amount'], errors='coerce')
    qty_num = pd.to_numeric(items['quantity'], errors='coerce')
    price_num = pd.to_numeric(products['unit_price'], errors='coerce')

    log_file.write(f"Orders with negative sales: {(sales_num < 0).sum()}\n")
    log_file.write(f"Orders with negative taxes: {(tax_num < 0).sum()}\n")
    log_file.write(f"Orders where discount > sale: {(discount_num > sales_num).sum()}\n")
    log_file.write(f"Items with 0 or negative quantity: {(qty_num <= 0).sum()}\n")
    log_file.write(f"Products with 0 or negative price: {(price_num <= 0).sum()}\n")

    # Saves cleaned numeric columns separatel so it dont overwrite the originals
    orders['sales_amount_num'] = sales_num
    items['quantity_num'] = qty_num
    products['unit_price_num'] = price_num

    check_outliers_iqr(orders, 'sales_amount_num', 'Orders', log_file)
    check_outliers_iqr(items, 'quantity_num', 'Order Items', log_file)
    check_outliers_iqr(products, 'unit_price_num', 'Products', log_file)

if __name__ == "__main__":
    print("Loading data")
    customers_df = pd.read_csv(os.path.join(data_dir, "customers_large.csv"))
    orders_df = pd.read_csv(os.path.join(data_dir, "orders_large.csv"))
    products_df = pd.read_csv(os.path.join(data_dir, "products_large.csv"))
    items_df = pd.read_csv(os.path.join(data_dir, "order_items_large.csv"))

    print("Running profiling")
    cust_profile = profile_dataset(customers_df, "customers_large.csv")
    ord_profile = profile_dataset(orders_df, "orders_large.csv")
    prod_profile = profile_dataset(products_df, "products_large.csv")
    items_profile = profile_dataset(items_df, "order_items_large.csv")

    cust_profile.to_csv(os.path.join(output_dir, "customers_profile.csv"), index=False)
    ord_profile.to_csv(os.path.join(output_dir, "orders_profile.csv"), index=False)
    prod_profile.to_csv(os.path.join(output_dir, "products_profile.csv"), index=False)
    items_profile.to_csv(os.path.join(output_dir, "items_profile.csv"), index=False)
    print("Exported profiling tables to CSV")

    # Writes all data quality findings to a text file
    print("Running data quality checks")
    with open(os.path.join(output_dir, "phase1_findings.txt"), "w") as log:
        log.write("--- Duplicate Records ---\n")
        check_duplicates(customers_df, "customer_id", "Customers", log)
        check_duplicates(orders_df, "order_id", "Orders", log)
        check_duplicates(products_df, "product_id", "Products", log)
        check_duplicates(items_df, "order_item_id", "Order Items", log)

        log.write("\n--- Integrity Constraints ---\n")
        check_foreign_keys(customers_df, "customer_id", orders_df, "customer_id", "Orders -> Customers", log)
        check_foreign_keys(orders_df, "order_id", items_df, "order_id", "Items -> Orders", log)
        check_foreign_keys(products_df, "product_id", items_df, "product_id", "Items -> Products", log)

        check_business_rules(customers_df, orders_df, items_df, products_df, log)
