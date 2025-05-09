import pandas as pd 
import numpy as np

def fix_swapped_columns(df:pd.DataFrame, column_order_old: list,column_order_new: list, swap_condition)->pd.DataFrame:
    """
    This function takes in the df_training and df_testing dataframes
    and fixes the swapped column entries.

    """
    df.loc[swap_condition, column_order_old] = df.loc[swap_condition, column_order_new].values

    return df