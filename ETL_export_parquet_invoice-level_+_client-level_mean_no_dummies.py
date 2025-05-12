# This file imports the original energy client + invoice data (train + final_test),
# and performs transformations
# (does not perform scaling or normalisation)
# the resulting tables are exported as parquet files to the data folder
# these can be imported into notebooks as needed

import pandas as pd
import pyarrow
import fastparquet
import datetime as dt

# Define data types for CSV imports

dtypes_client_train = {
    "disrict": "str",
    "client_id": "str",
    "client_catg": "str",
    "region": "str",
    "target": "int",
}

date_format_client = {"creation_date": "%d/%m/%Y"}

dtypes_client_test = {
    "disrict": "str",
    "client_id": "str",
    "client_catg": "str",
    "region": "str",
}

dtypes_invoice = {
    "client_id": "str",
    "tarif_type": "str",
    "counter_number": "str",
    "counter_statue": "str",
    "counter_code": "str",
    "reading_remarque": "int",
    "counter_coefficient": "float",
    "consommation_level_1": "float",
    "consommation_level_2": "float",
    "consommation_level_3": "float",
    "consommation_level_4": "float",
    "old_index": "float",
    "new_index": "float",
    "months_number": "int",
    "counter_type": "str",
}

date_format_invoice = {
    "invoice_date": "%Y-%m-%d",
}

# Import CSVs

client_train = pd.read_csv(
    "data/client_train.csv",
    dtype=dtypes_client_train,
    parse_dates=["creation_date"],
    date_format=date_format_client,
    low_memory=False,
)

invoice_train = pd.read_csv(
    "data/invoice_train.csv",
    dtype=dtypes_invoice,
    parse_dates=["invoice_date"],
    date_format=date_format_invoice,
    low_memory=False,
)

client_final_test = pd.read_csv(
    "data/client_test.csv",
    dtype=dtypes_client_test,
    parse_dates=["creation_date"],
    date_format=date_format_client,
    low_memory=False,
)

invoice_final_test = pd.read_csv(
    "data/invoice_test.csv",
    dtype=dtypes_invoice,
    parse_dates=["invoice_date"],
    date_format=date_format_invoice,
    low_memory=False,
)


# Remove incorrect data / from wrong time period
def data_cleaning(df):

    # First convert 'A' to a number (for example 99) or drop those rows
    df["counter_statue"] = df["counter_statue"].replace("A", 99)

    # Then convert the column to integer type
    df["counter_statue"] = df["counter_statue"].astype(int)

    # Now you can filter as before
    df = df[~(df["counter_statue"] > 5)]

    # drop months number >12
    df = df[df["months_number"] <= 12]

    # drop invoice dates before 2005
    df = df[df["invoice_date"] >= pd.to_datetime("2005-01-01", format="%Y-%m-%d")]

    # drop the counter number, there are a lot of 0 values in both train and test data
    df = df.drop("counter_number", axis=1)

    return df

def transform_df_and_export_parquet(
    df_client, df_invoice, export_path_agg, export_path_non_agg
):

    ## DATA CLEANING (date ranges, missing values)

    df_invoice_clean = data_cleaning(df_invoice)

    ## CLIENT DATA TRANSFORM ##

    ## Drop rows from client table where invoice data has been cleaned
    client_ids_in_df_invoice_clean = list(set(df_invoice_clean["client_id"]))
    df_client = df_client[df_client["client_id"].isin(client_ids_in_df_invoice_clean)]

    import data_aggregation_function as daf
    """
    # Set target variables to labels instead of integers
    target_map = {0:"Not Fraud", 1:"Fraud"}

    if "target" in df_client.columns:
        df_client["target"] = df_client["target"].map(target_map)
    """
    # Convert client creation_date to integer so model can understand (this is Excel format)
    df_client["creation_date"] = (
        pd.to_datetime(df_client["creation_date"]) - pd.Timestamp("1900-01-01")
    ).dt.days + 2

    ## INVOICE DATA TRANSFORM ##

    # Create dummy columns for counter_type values (GAZ/ELEC)
    categorical_cols_pre_agg = ["counter_type"]

    df_invoice_aggregated_with_1st_dummies = daf.get_dummies_and_remerge(
        df_invoice_clean, categorical_cols_pre_agg
    )

    # Set aggregations to use for each invoice column when grouping by client-id
    cols_and_aggs = {
        "invoice_date": {
            "is_active": False,
            "aggregation": "mean",
        },
        "tarif_type": {
            "is_active": True,
            "aggregation": "safe_mode",
        },
        "counter_number": {
            "is_active": False,
            "aggregation": "nunique",
            "comment": "# to count the number of different counters per client - deactivated because we have lots of 0s",
        },
        "counter_statue": {
            "is_active": True,
            "aggregation": "safe_mode",
            "comment": "# can be different values for different counters",
        },
        "counter_code": {
            "is_active": True,
            "aggregation": "safe_mode",
            "comment": "# another option would be using dummy values to see if category present for customer. We do not know what this means",
        },
        "reading_remarque": {
            "is_active": True,
            "aggregation": "max",
        },
        "counter_coefficient": {
            "is_active": True,
            "aggregation": "safe_mode",
        },
        "consommation_level_1": {
            "is_active": True,
            "aggregation": "mean",
        },
        "consommation_level_2": {
            "is_active": True,
            "aggregation": "mean",
        },
        "consommation_level_3": {
            "is_active": True,
            "aggregation": "mean",
        },
        "consommation_level_4": {
            "is_active": True,
            "aggregation": "mean",
        },
        "old_index": {
            "is_active": False,
            "aggregation": "mean",
            "comment": "# we drop these because of correlation to each other and we will use it for feature engineering later",
        },
        "new_index": {
            "is_active": False,
            "aggregation": "mean",
        },
        "months_number": {
            "is_active": False,
            "aggregation": "safe_mode",
        },
        "counter_type": {
            "is_active": False,
            "aggregation": "safe_mode",
            "comment": "# gaz/elec - keep as dummies",
        },
        "counter_type_GAZ": {
            "is_active": True,
            "aggregation": "max",
            "comment": "",
        },
        "counter_type_ELEC": {
            "is_active": True,
            "aggregation": "max",
            "comment": "",
        },
        "invoice_diff": {
            "is_active": False,
            "aggregation": "mean",
            "comment": "# feature engineering",
        },
        "meter_broken": {
            "is_active": False,
            "aggregation": "safe_mode",
            "comment": "# feature engineering",
        },
    }

    cols_to_drop = [
        "invoice_date",
        "old_index",
        "new_index",
        #'counter_type',
        #'invoice_diff',
        #'meter_broken'
    ]

    # Create aggregated invoice data and drop unneeded columns
    df_invoice_aggregated = daf.aggregate_df(
        df_invoice_aggregated_with_1st_dummies, cols_to_drop, cols_and_aggs
    )

    # For non-aggregated version, drop unneeded columns
    df_invoice_non_agg = df_invoice_aggregated_with_1st_dummies.loc[
        :, ~df_invoice_aggregated_with_1st_dummies.columns.isin(cols_to_drop)
    ]

    ## MERGE CLIENT AND INVOICE DATA

    # Combine aggregated invoice data with client data
    df_client_merged_w_agg_invoice = pd.merge(
        df_client, df_invoice_aggregated, how="left", on="client_id"
    )

    # Combine non-aggregated invoice data with client data
    df_client_merged_w_non_agg_invoice = pd.merge(
        df_client, df_invoice_non_agg, how="left", on="client_id"
    )

    categorical_cols_after_agg = [
        "disrict",
        "client_catg",
        "region",
        "tarif_type",
        "reading_remarque",
        "counter_statue",
        "counter_code",
    ]

    # Drop client_id as no longer needed for grouping (and can cause leakage)
    cols_to_drop_at_end = [
        "client_id",
    ]

    # drop cols listed above
    df_agg_to_export = df_client_merged_w_agg_invoice.loc[
        :, ~df_client_merged_w_agg_invoice.columns.isin(cols_to_drop_at_end)
    ]

    df_non_agg_to_export = df_client_merged_w_non_agg_invoice.loc[
        :, ~df_client_merged_w_non_agg_invoice.columns.isin(cols_to_drop_at_end)
    ]

    ## EXPORT TO PARQUET FILES

    # Export transformed, client-level aggregated data as Parquet, ready for import into notebooks
    df_agg_to_export.to_parquet(export_path_agg)

    # Export transformed, invoice-level 'non-aggregated' data as Parquet, ready for import into notebooks
    df_non_agg_to_export.to_parquet(export_path_non_agg)


## RUN TRANSFORM FUNCTION ON OUR DATA

transform_df_and_export_parquet(
    client_train,
    invoice_train,
    "data/df_train_no_dummies_agg.parquet",
    "data/df_train_no_dummies_non_agg.parquet",
)
transform_df_and_export_parquet(
    client_final_test,
    invoice_final_test,
    "data/df_final_test_no_dummies_agg.parquet",
    "data/df_final_test_no_dummies_non_agg.parquet",
)
