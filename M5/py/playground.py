import pickle
import os
import pandas as pd

base_path = 'M5/data'

with open(os.path.join(base_path, 'grouped_df.pickle'), 'rb') as fp:
    df_list = pickle.load(fp)


df_prod = df_list[0]
df_prod['sum'] = pd.to_numeric(df_prod['sum'])

top_20 = df_prod.nlargest(20, columns='sum')
