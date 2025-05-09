# check if sum of the consommation_level values is equal to the difference between old index and new index
# Create a mask for the condition

import pandas as pd

def broken_meter_function(df:pd.DataFrame)->pd.DataFrame:
        mask = (df['consommation_level_1'] + 
                df['consommation_level_2'] + 
                df['consommation_level_3'] + 
                df['consommation_level_4']) != (df['new_index'] - df['old_index'])

        # Create a new row in the dataframe
        df['meter_broken'] = mask.astype(int)

        return df