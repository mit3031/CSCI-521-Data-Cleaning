import pandas as pd
import os

"""
Finds exact duplicates, ignoring id, and maps dropped id's to the surviving id
"""
def find_duplicate_mapping(df, id_col):
    # Gets all the columns except the primary key
    check_cols = [col for col in df.columns if col != id_col]
    
    # Group by all other columns to find duplicates
    grouped = df.groupby(check_cols, dropna=False)[id_col].apply(list)
    
    mapping = {}
    for id_list in grouped:
        if len(id_list) > 1:
            surviving_id = id_list[0]
            # Map all subsequent dropped ID's to the first one
            for dropped_id in id_list[1:]:
                mapping[dropped_id] = surviving_id
                
    return mapping

"""
Drops exact duplicates, ignoring id, and prints the count for the report
"""
def apply_dedup(df, id_col):
    check_cols = [col for col in df.columns if col != id_col]
    rows_before = len(df)
    
    df_clean = df.drop_duplicates(subset=check_cols, keep='first').copy()
    
    rows_after = len(df_clean)
    dropped_count = rows_before - rows_after
    print(f"Dropped {dropped_count} duplicate rows")
    
    return df_clean

"""
Updates foreign keys in the child table using the mapping dictionary
"""
def remap_children(child_df, fk_col, mapping):

    child_df_clean = child_df.copy()
    # Replaces old dropped id's with the surviving id
    child_df_clean[fk_col] = child_df_clean[fk_col].replace(mapping)
    return child_df_clean

"""
Adds a -1 placeholder customer and updates orphan orders
"""
def add_unknown_customer(customers_df, orders_df):
    
    valid_customers = customers_df['customer_id'].unique()
    
    orders_df_clean = orders_df.copy()
    # Finds orders where the customer_id is not in the valid list
    for index, row in orders_df_clean.iterrows():
        if row['customer_id'] not in valid_customers:
            orders_df_clean.at[index, 'customer_id'] = -1
    
    # Create the placeholder row. The state is set to NY so that it survives the dictionary mapping
    unknown_cust = {
        'customer_id': -1,
        'first_name': 'Unknown',
        'last_name': 'Unknown',
        'email': 'unknown@missing.local', 
        'phone': '0000000000',
        'city': 'Unknown',
        'state': 'NY',
        'zip_code': 0,
        'signup_date': '1900-01-01'
    }
    
    unknown_df = pd.DataFrame([unknown_cust])
    customers_df_clean = pd.concat([customers_df, unknown_df], ignore_index=True)
    
    return customers_df_clean, orders_df_clean

"""
Adds a -1 placeholder product and updates orphan order items
"""
def add_unknown_product(products_df, order_items_df):
    valid_products = products_df['product_id'].unique()
    
    order_items_df_clean = order_items_df.copy()
    orphan_mask = ~order_items_df_clean['product_id'].isin(valid_products)
    order_items_df_clean.loc[orphan_mask, 'product_id'] = -1
    
    unknown_prod = {
        'product_id': -1,
        'category': 'Unknown',
        'subcategory': 'Unknown',
        'brand': 'Unknown',
        'unit_price': 0.0 
    }
    
    unknown_df = pd.DataFrame([unknown_prod])
    products_df_clean = pd.concat([products_df, unknown_df], ignore_index=True)
    
    return products_df_clean, order_items_df_clean

"""Main runner script to assemble everything"""
def run_cleaning_pipeline(raw_dir, output_dir):
    print("Starting pipeline...")
    
    # Loads raw data
    customers = pd.read_csv(f"{raw_dir}/customers_large.csv")
    products = pd.read_csv(f"{raw_dir}/products_large.csv")
    orders = pd.read_csv(f"{raw_dir}/orders_large.csv")
    order_items = pd.read_csv(f"{raw_dir}/order_items_large.csv")

    # Role A here
    
    # Maps the duplicates before dropping them
    cust_mapping = find_duplicate_mapping(customers, 'customer_id')
    prod_mapping = find_duplicate_mapping(products, 'product_id')
    
    # Updates the child tables so they don't get orphaned
    orders = remap_children(orders, 'customer_id', cust_mapping)
    order_items = remap_children(order_items, 'product_id', prod_mapping)
    
    # Actually drop the duplicates now that children are safe
    print("Customers deduplication:")
    customers = apply_dedup(customers, 'customer_id')
    print("Products deduplication:")
    products = apply_dedup(products, 'product_id')
    print("Order Items deduplication:")
    order_items = apply_dedup(order_items, 'order_item_id')
    
    # Handles the original phase 1 orphans by routing them to ID -1
    customers, orders = add_unknown_customer(customers, orders)
    products, order_items = add_unknown_product(products, order_items)

    # Role B, C, D go Here

    # Make all the new and improved table files and puts it into the data folder
    os.makedirs(output_dir, exist_ok=True)
        
    customers.to_csv(f"{output_dir}/customers_master.csv", index=False)
    products.to_csv(f"{output_dir}/products_master.csv", index=False)
    orders.to_csv(f"{output_dir}/orders_master.csv", index=False)
    order_items.to_csv(f"{output_dir}/order_items_master.csv", index=False)
    
    print("Pipeline complete. Clean master files saved.")
    
    return customers, orders, products, order_items

if __name__ == "__main__":
    cust, ords, prods, items = run_cleaning_pipeline('data', 'output')
    # print(f"Final customer count for report validation: {len(cust)}")

    orders_to_unknown = len(ords[ords['customer_id'] == -1])
    items_to_unknown = len(items[items['product_id'] == -1])
    print(f"Orders mapped to Unknown (-1): {orders_to_unknown}")
    print(f"Items mapped to Unknown (-1): {items_to_unknown}")

    cust_orphans = len(ords[~ords['customer_id'].isin(cust['customer_id'])])
    prod_orphans = len(items[~items['product_id'].isin(prods['product_id'])])
    print(f"Remaining Customer Orphans: {cust_orphans} (Should be 0)")
    print(f"Remaining Product Orphans: {prod_orphans} (Should be 0)")