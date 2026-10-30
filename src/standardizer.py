import pandas as pd
import numpy as np

def standardize_states(customers_df):
    customers_df["state"] = customers_df["state"].astype("string").str.strip().str.upper()
    customers_df["state"] = customers_df["state"].replace({
        "N.Y.": "NY",
        "NEW YORK": "NY"
    })
    return customers_df


def standardize_phones(customers_df):
    customers_df["phone"] = customers_df["phone"].astype("string").str.replace(r"\D", "", regex=True)
    return customers_df


def standardize_categories(products_df):
    products_df["category"] = products_df["category"].astype("string").str.strip().str.title()
    products_df["category"] = products_df["category"].replace("Electornics", "Electronics")

    products_df["subcategory"] = products_df["subcategory"].astype("string").str.strip().str.title()
    products_df["brand"] = products_df["brand"].astype("string").str.strip()

    return products_df


def standardize_text(customers_df):
    for col in ["first_name", "last_name", "city"]:
        customers_df[col] = customers_df[col].astype("string").str.strip().str.title()

    customers_df["email"] = customers_df["email"].astype("string").str.strip().str.lower()

    return customers_df
