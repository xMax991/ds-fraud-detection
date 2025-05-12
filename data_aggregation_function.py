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

def get_dummies_and_remerge(df, cols_for_get_dummies):
    """ Convert specified columns into dummy columns

    Args:
        df (dataframe): _description_
        cols_for_get_dummies (list): _description_

    Returns:
        dataframe 
    """
    # Ensure all columns in cols_for_get_dummies exist in the DataFrame
    cols_for_get_dummies = [col for col in cols_for_get_dummies if col in df.columns]

    df_new = pd.concat(
        [
            df.loc[:, ~df.columns.isin(cols_for_get_dummies)],
            pd.get_dummies(df[cols_for_get_dummies]),
        ],
        axis=1,
    )

    return df_new


# Custom function to get the most common value safely
def safe_mode(x):
    if x.empty:
        return None
    mode_vals = x.mode()
    return mode_vals.iloc[0] if not mode_vals.empty else None


def aggregate_df(df, cols_to_drop, cols_and_aggs):
    """aggregate dataframe using supplied dictionary

    Args:
        df (dataframe): Pandas Dataframe
        cols_to_drop (list): List of columns to drop.
        cols_and_aggs (dict): Dict with columns as entries, each with a dict defining:
            'is_active': True/False
            'aggregation': 'mean', 'max', 'mode', 'nunique'
            'comment':
    """
    df = df.loc[:, ~df.columns.isin(cols_to_drop)]

    aggs = {}

    for column, attributes in cols_and_aggs.items():
        if attributes.get("is_active") == True:
            agg_mode = attributes.get("aggregation")
            if agg_mode == "safe_mode":
                aggs[column] = safe_mode
            else:
                aggs[column] = agg_mode
        else:
            continue

    agg_trans = df.groupby(["client_id"]).agg(aggs)
    # agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = df.groupby("client_id").size().reset_index(name="transactions_count")
    return pd.merge(df, agg_trans, on="client_id", how="left")


"""def aggregate_by_client_id(invoice_data):

    # Custom function to get the most common value safely
    def safe_mode(x):
        if x.empty:
            return None
        mode_vals = x.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else None
    
    aggs = {}
    aggs['disrict'] 			 = [safe_mode]
    #aggs['client_id'] 			 = ['mean']
    aggs['client_catg'] 	     = [safe_mode]
    aggs['region'] 			     = [safe_mode]
    aggs['creation_date'] 		 = [safe_mode]
    aggs['target'] 			     = [safe_mode]
    #aggs['invoice_date'] 	     = ['mean']
    aggs['tarif_type'] 			 = [safe_mode]
    #aggs['counter_number'] 	 = ['nunique'] # to count the number of different counters per client
    aggs['counter_statue'] 	     = [safe_mode] # can be different values for different counters
    aggs['counter_code'] 	     = [safe_mode] # another option would be using dummy values to see if category present for customer. We do not know what this means
    aggs['reading_remarque'] 	 = ['max']
    aggs['counter_coefficient']  = [safe_mode]
    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']
    #aggs['old_index'] 			 = ['mean'] # we drop these because of correlation to each other and we will use it for feature engineering later
    #aggs['new_index'] 			 = ['mean']
    aggs['months_number']   	 = [safe_mode]
    #aggs['counter_type']   	 = [safe_mode] # gaz/elec - keep as dummies
    #aggs['invoice_diff']   	 = ['mean'] # feature engineering
    #aggs['meter_broken']   	 = [safe_mode] 

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='{}transactions_count'.format('1')))
    return pd.merge(df, agg_trans, on='client_id', how='left')
    """
