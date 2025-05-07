import pandas as pd
import numpy as np

def data_cleaning(df):

    # First convert 'A' to a number (for example 99) or drop those rows
    df['counter_statue'] = df['counter_statue'].replace('A', 99)
    # Then convert the column to integer type
    df['counter_statue'] = df['counter_statue'].astype(int)
    # Now you can filter as before
    df = df[~(df['counter_statue'] > 5)]
    # drop months number >12
    df = df[df['months_number'] <= 12]
    # covert datatypes ino date time
    df.creation_date = pd.to_datetime(df.creation_date, format='%d/%m/%Y')
    df.invoice_date = pd.to_datetime(df.invoice_date, format='%Y-%m-%d')
    # drop invoice dates before 2005
    df = df[df['invoice_date'] > pd.to_datetime('2005-01-01', format='%Y-%m-%d')]
    return df