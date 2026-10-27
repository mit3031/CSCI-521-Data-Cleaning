import os
import pandas as pd

for file in ["customers_large.csv", "orders_large.csv", "products_large.csv", "order_items_large.csv"]:
    path = os.path.join("data", file)
    size_kb = os.path.getsize(path) / 1024
    rows = len(pd.read_csv(path))
    
    if size_kb > 1024:
        print(f"{file}: {size_kb/1024:.2f} MB | {rows:,} rows")
    else:
        print(f"{file}: {size_kb:.2f} KB | {rows:,} rows")