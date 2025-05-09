# check if sum of the consommation_level values is equal to the difference between old index and new index
# Create a mask for the condition
mask = (df_train['consommation_level_1'] + 
        df_train['consommation_level_2'] + 
        df_train['consommation_level_3'] + 
        df_train['consommation_level_4']) != (df_train['new_index'] - df_train['old_index'])

# Create a new row in the dataframe
df_train['meter_broken'] = mask.astype(int)