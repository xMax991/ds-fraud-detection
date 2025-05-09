def aggregate_by_client_id(invoice_data):
    aggs = {}
    aggs['disrict'] 			 = ['mode']
    #aggs['client_id'] 			 = ['mean']
    aggs['client_catg'] 	     = ['mode']
    aggs['region'] 			     = ['mode']
    aggs['creation_date'] 		 = ['mode']
    aggs['target'] 			     = ['mode']
    #aggs['invoice_date'] 	     = ['mean']
    aggs['tarif_type'] 			 = ['mode']
    #aggs['counter_number'] 	     = ['nunique'] # to count the number of different counters per client
    aggs['counter_statue'] 	     = ['mode'] # can be different values for different counters
    aggs['counter_code'] 	     = ['mode'] # another option would be using dummy values to see if category present for customer. We do not know what this means
    aggs['reading_remarque'] 	 = ['mode']
    aggs['counter_coefficient'] 	= ['mode']
    aggs['consommation_level_1'] 	= ['mean']
    aggs['consommation_level_2'] 	= ['mean']
    aggs['consommation_level_3'] 	= ['mean']
    aggs['consommation_level_4'] 	= ['mean']
    #aggs['old_index'] 			 = ['mean'] # we drop these because of correlation to each other and we will use it for feature engineering later
    #aggs['new_index'] 			 = ['mean']
    aggs['months_number'] 			 = ['mode']
    aggs['counter_type'] 			 = ['mode']
    #aggs['invoice_diff'] 			 = ['mean'] # feature engineering
    aggs['meter_broken'] 			 = ['mode'] 

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='{}transactions_count'.format('1')))
    return pd.merge(df, agg_trans, on='client_id', how='left')