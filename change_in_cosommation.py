import pandas as pd 

def change_in_consommation(df:pd.DataFrame)->pd.DataFrame:

"""
calculating the biggest change in consumption per costumer and consumption level
"""

    df_sorted = df.sort_values(['client_id', 'invoice_date'], ascending=[True, False])
    
    consommation_cols = ['consommation_level_1','consommation_level_2','consommation_level_3','consommation_level_4']
    df_sorted[consommation_cols] = df_sorted.groupby('client_id')[consommation_cols].diff().abs()

    max_changes = df_sorted.groupby('client_id')[consommation_cols].max().reset_index()

    return max_changes