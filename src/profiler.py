import pandas as pd

def profile_dataset(file_path):
    
    df = pd.read_csv(file_path)

    # Record count
    total_records = len(df)

    # Attribute count
    total_attributes = len(df.columns)

    print(f"Profiling Report: {file_path}")
    print(f"Total Records (Rows): {total_records}")
    print(f"Total Attributes (Columns): {total_attributes}")
    print("-" * 40)

    # list that stores results for each attribute
    stats_list = []

    # goes through all attributes
    for col in df.columns:

        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / total_records) * 100

        # cardinality
        unique_count = df[col].nunique()

        # Summary Stats

        # Data type
        col_type = df[col].dtype

        # Mode
        mode_series = df[col].mode()

        if len(mode_series) > 0:
            col_mode = mode_series[0]
        else:
            col_mode = "N/A"

        # default values for stats that don't apply
        col_mean = "N/A"
        col_median = "N/A"
        col_min = "N/A"
        col_max = "N/A"

        # Calculates stats for numeric attributes
        if col_type == "int64" or col_type == "float64":
            col_mean = round(df[col].mean(), 2)
            col_median = round(df[col].median(), 2)
            col_min = df[col].min()
            col_max = df[col].max()

        else:
            # Try to get minimum and maximum for non numeric attributes
            try:
                col_min = df[col].min()
                col_max = df[col].max()
            except:
                pass

        stats_list.append({
            "Attribute": col,
            "Data Type": col_type,
            "Missing Count": missing_count,
            "Missing %": round(missing_percentage, 2),
            "Unique Values": unique_count,
            "Mode": col_mode,
            "Mean": col_mean,
            "Median": col_median,
            "Min": col_min,
            "Max": col_max
        })

    summary_table = pd.DataFrame(stats_list)

    return summary_table

#Test example on how to run it on the csv
#customers = profile_dataset("data/customers_large.csv")
#print(customers)