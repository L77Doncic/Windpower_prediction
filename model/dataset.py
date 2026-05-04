import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def data_preprocess(df):
    df = df.sort_values(by='DATATIME', ascending=True)
    print('df.shape:', df.shape)
    print(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    print('After Dropping duplicates:', df.shape)
    if df['YD15'].isnull().all():
        df['YD15'] = df["ROUND(A.POWER,0)"]
    columns = ['ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        outliers_index = df[outliers].index
        df.loc[outliers_index, col] = np.nan
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df.set_index('DATATIME', inplace=True)
    df = df.resample(rule='15min', label='right', closed='right').interpolate(method='linear',
                                                                            limit_direction='both').reset_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df.set_index('DATATIME', inplace=True)
    df = df.interpolate(method='time', limit_direction='both').reset_index()
    columns = ['ROUND(A.POWER,0)', 'YD15']
    for col in columns:
        if df['DATATIME'].dtype != 'datetime64[ns]':
            df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df['DATE'] = df['DATATIME'].dt.date
        grouped = df.groupby(['DATE'])
        group_sizes = grouped.size()
        unique_counts = grouped[col].nunique()
        stale_rows = unique_counts[unique_counts == 1].index.tolist()
        stale_indices = []
        for date in stale_rows:
            indices = df.index[(df['DATE'] == date)]
            stale_indices.extend(indices.tolist())
        df.loc[stale_indices, col] = np.nan
        df.drop(columns=['DATE'], inplace=True)
    df.loc[df['ROUND(A.WS,1)'] < 0, 'ROUND(A.WS,1)'] = np.nan
    df.loc[df['YD15'] > 1e10, 'YD15'] = np.nan
    df.loc[df['ROUND(A.POWER,0)'] > 1e10, 'ROUND(A.POWER,0)'] = np.nan
    df.loc[df['YD15'] < -1e10, 'YD15'] = np.nan
    df.loc[df['ROUND(A.POWER,0)'] < -1e10, 'ROUND(A.POWER,0)'] = np.nan
    for col in columns:
        df.loc[(df['ROUND(A.WS,1)'] == 0) & (df[col] > 0), col] = 0
        df.loc[(df['ROUND(A.WS,1)'] > 20) & (df[col] != 0), col] = 0
        df.loc[(df['ROUND(A.WS,1)'] > 5) & (df[col] == 0), col] = np.nan
    print('After Resampling:', df.shape)
    X = df[['WINDSPEED', 'PREPOWER', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']]
    imputer = KNNImputer(n_neighbors=50)
    filled_samples = imputer.fit_transform(X)
    df.loc[X.index, ['WINDSPEED', 'PREPOWER', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']] = filled_samples
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df.set_index('DATATIME', inplace=True)
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    return df

def feature_engineer(df):
    # Implement cyclical encoding as described in the paper (Section 3.2)
    df['month_sin'] = np.sin(2 * np.pi * df['DATATIME'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['DATATIME'].dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['DATATIME'].dt.day / 31) # Approximate with 31
    df['day_cos'] = np.cos(2 * np.pi * df['DATATIME'].dt.day / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['DATATIME'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['DATATIME'].dt.hour / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['DATATIME'].dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['DATATIME'].dt.minute / 60)
    return df