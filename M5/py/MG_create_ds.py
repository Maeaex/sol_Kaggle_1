# 1 https://www.kaggle.com/kernelgenerator/m5-accuracy-tweedie-is-back
# 2 https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration
# 3 https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda
# 4 https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models
# https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic


# De-noising: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.565.1807&rep=rep1&type=pdf
# --> https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html

# Croston' Method
# --> https://pypi.org/project/croston/

import pandas as pd
import os
import gc


base_path = '../data'

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

def create_sales():
    dtypes_sales = {"id": "category", "item_id": "category", "dept_id": "category", "cat_id": "category",
                    "store_id": "category", "state_id": "category"}
    for i in range(1, 1913):
        dtypes_sales.update({f"d_{i}": "int16"})

    df_sales = pd.read_csv(os.path.join(base_path, 'sales_train_validation.csv'), dtype=dtypes_sales)
    df_sales = pd.melt(df_sales, id_vars=df_sales.columns.to_list()[:6], value_vars=df_sales.columns.to_list()[6:],
                       var_name='d', value_name='sales')

    return df_sales

def create_cal():
    dtypes_cal = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
                  "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                  "month": "int16", "year": "int16", "snap_CA": "int8", 'snap_TX': 'int8', 'snap_WI': 'int8'}

    df_cal = pd.read_csv(os.path.join(base_path, 'calendar.csv'), dtype=dtypes_cal)
    cols = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    df_day = pd.get_dummies(df_cal['wday'])
    df_day.columns = cols
    df_cal = pd.concat([df_cal, df_day.set_index(df_cal.index)], axis=1)
    df_cal = df_cal.drop(['weekday', 'wday'], axis=1)

    return df_cal

# Define dtypes for sales_train_validation.csv


dtypes_price = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
df_price = pd.read_csv(os.path.join(base_path,'sell_prices.csv'), dtype=dtypes_price)
df_sales = create_sales()
df_cal = create_cal()

df = df_sales.merge(df_cal, left_on='d', right_on='d', copy=False)
df = df.merge(df_price, left_on=['item_id', 'store_id', 'wm_yr_wk'], right_on=['item_id', 'store_id', 'wm_yr_wk'], copy=False)
df['revenue'] = df['sell_price'] * df['sales']
create_fea(df)

df = df.drop(columns=['wm_yr_wk', 'state_id', 'event_name_2', 'event_type_2'])


pd.DataFrame.to_feather(df, os.path.join(base_path, 'master_df.ftr'))

gc.collect()