"""Role D: flag numeric anomalies and add Min-Max normalized metrics.

All functions return copies and preserve original values, rows, and indexes.
Numeric strings are accepted. Missing, unparseable, and infinite values are
unevaluated: their rule flags are False and their normalized values are NaN.
A False flag therefore does not guarantee that a value is valid or present;
missing-value treatment remains a separate cleaning responsibility.

The integration pipeline reserves product_id == -1 for an unknown product.
Its placeholder unit_price is not a real measurement and is excluded from
price flags, outlier statistics, and normalization (normalized price is NaN).
"""

import numpy as np
import pandas as pd


def _validate_columns(df, columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique")
    if isinstance(columns, str):
        columns = [columns]
    try:
        columns = list(columns)
    except TypeError as exc:
        raise TypeError("columns must be a column name or iterable of names") from exc
    if any(not isinstance(column, str) for column in columns):
        raise TypeError("column names must be strings")
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return columns


def _numeric_values(df, column):
    source = df[column]
    if (pd.api.types.is_datetime64_any_dtype(source.dtype)
            or pd.api.types.is_timedelta64_dtype(source.dtype)
            or pd.api.types.is_bool_dtype(source.dtype)):
        raise TypeError(f"{column!r} must contain real numbers or numeric strings")
    values = pd.to_numeric(source, errors="coerce")
    if pd.api.types.is_complex_dtype(values.dtype):
        raise TypeError(f"{column!r} must contain real numbers or numeric strings")
    values = values.where(np.isfinite(values))
    if column == "unit_price" and "product_id" in df.columns:
        unknown = pd.to_numeric(df["product_id"], errors="coerce").eq(-1)
        values = values.mask(unknown.fillna(False))
    return values


def _iqr_bounds(values):
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def flag_invalid_quantities_and_prices(order_items_df, products_df):
    """Flag quantity <= 0 and unit_price <= 0 without changing either value.

    Returns (order_items_copy, products_copy). Unknown product ID -1 is exempt
    from price validation; actual products with a zero price are still flagged.
    """
    _validate_columns(order_items_df, ["quantity"])
    _validate_columns(products_df, ["unit_price"])
    items = order_items_df.copy()
    products = products_df.copy()
    quantities = _numeric_values(order_items_df, "quantity")
    prices = _numeric_values(products_df, "unit_price")
    items["non_positive_qty_flag"] = quantities.le(0).fillna(False).astype(bool)
    products["non_positive_price_flag"] = prices.le(0).fillna(False).astype(bool)
    return items, products


def flag_financial_anomalies(orders_df):
    """Flag negative sales/tax and discount > sale using signed comparisons.

    Negative values may be returns, so do not take absolute values, delete
    records, or replace the amounts. A discount comparison requires both
    amounts to be finite numeric values.
    """
    _validate_columns(orders_df, ["sales_amount", "tax_amount", "discount"])
    orders = orders_df.copy()
    sales = _numeric_values(orders_df, "sales_amount")
    tax = _numeric_values(orders_df, "tax_amount")
    discount = _numeric_values(orders_df, "discount")
    orders["negative_sales_flag"] = sales.lt(0).fillna(False).astype(bool)
    orders["negative_tax_flag"] = tax.lt(0).fillna(False).astype(bool)
    exceeds = sales.notna() & discount.notna() & discount.gt(sales)
    orders["discount_exceeds_sale_flag"] = exceeds.fillna(False).astype(bool)
    return orders


def flag_outliers(df, numeric_cols, *, reference_df=None):
    """Add is_outlier if ANY selected value is outside its 1.5-IQR fences.

    Use Q1/Q3 with pandas' default linear interpolation, matching Phase 1.
    Boundary values are not outliers. For zero IQR, values differing from the
    common quartile are outliers. Empty/all-missing columns cannot flag rows.
    No rows are deleted: large orders and expensive products may be legitimate.

    By default the input supplies the reference distribution. Pass an optional
    reference_df to retain the raw Phase 1 distribution after deduplication or
    placeholder insertion. Existing is_outlier values are replaced, allowing
    reproducible reruns. An empty numeric_cols produces an all-False flag.
    """
    columns = _validate_columns(df, numeric_cols)
    reference = df if reference_df is None else reference_df
    _validate_columns(reference, columns)
    result = df.copy()
    outliers = pd.Series(False, index=df.index, dtype=bool)
    for column in columns:
        values = _numeric_values(df, column)
        baseline = _numeric_values(reference, column).dropna()
        if baseline.empty:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lower, upper = _iqr_bounds(baseline)
            if not np.isfinite([lower, upper]).all():
                # An equivalent scaled comparison avoids overflow for very
                # large finite values, even when an IQR fence exceeds float64.
                scale = baseline.astype("float64").abs().max()
                lower, upper = _iqr_bounds(baseline / scale)
                values = values / scale
            outside = values.lt(lower) | values.gt(upper)
        outliers |= outside.fillna(False).astype(bool)
    result["is_outlier"] = outliers
    return result


def normalize_metrics(df, numeric_cols):
    """Add <column>_normalized = (x - min) / (max - min) in [0, 1].

    Fit each selected metric independently using finite values, including
    negatives and outliers; original amounts are never overwritten. Constant
    columns map finite values to 0.0. Empty/all-invalid columns remain NaN.
    Empty numeric_cols returns an unchanged copy. Repeated calls overwrite the
    same derived columns rather than creating duplicates.
    """
    columns = _validate_columns(df, numeric_cols)
    result = df.copy()
    for column in columns:
        values = _numeric_values(df, column)
        normalized = pd.Series(np.nan, index=df.index, dtype="float64")
        if values.notna().any():
            minimum, maximum = values.min(), values.max()
            if minimum == maximum:
                normalized.loc[values.notna()] = 0.0
            elif pd.api.types.is_integer_dtype(values.dtype):
                # Center integers exactly before conversion, avoiding signed
                # overflow and loss of small differences above 2**53.
                offsets = values.astype(object) - int(minimum)
                normalized = pd.to_numeric(offsets, errors="coerce").astype("float64")
                normalized = normalized / (int(maximum) - int(minimum))
            else:
                values = values.astype("float64")
                minimum, maximum = float(minimum), float(maximum)
                span = maximum - minimum
                if np.isfinite(span):
                    normalized = (values - minimum) / span
                else:
                    # Halve first when max - min would overflow float64.
                    normalized = (values / 2 - minimum / 2) / (maximum / 2 - minimum / 2)
        result[f"{column}_normalized"] = normalized.clip(0.0, 1.0)
    return result
