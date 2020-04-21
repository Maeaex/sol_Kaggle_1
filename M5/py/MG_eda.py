import numpy as np
import pandas as pd
import os
import pickle

base_path = '../data'
df = pd.read_feather('M5/data/master_df.ftr')



# Get importance of features by groupings based on weigthed RMSSE loss function
df_prod = pd.DataFrame({'sum': df.groupby(['item_id'])["revenue"]}).reset_index()
df_dept = pd.DataFrame({'sum': df.groupby(['dept_id'])["revenue"]}).reset_index()
df_cat = pd.DataFrame({'sum': df.groupby(['cat_id'])["revenue"]}).reset_index()

del df




'''

df_pickle = [df_prod, df_dept, df_cat]

with open(os.path.join(base_path, 'grouped_df.pickle'), 'wb') as fp:
    pickle.dump(df_pickle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
pd.DataFrame.to_feather(df_prod, os.path.join(base_path, 'revenue_by_prod.ftr'))
pd.DataFrame.to_feather(df_dept, os.path.join(base_path, 'revenue_by_dept.ftr'))
pd.DataFrame.to_feather(df_cat, os.path.join(base_path, 'revenue_by_cat.ftr'))

--> pyarrow.lib.ArrowInvalid: Message
'''



