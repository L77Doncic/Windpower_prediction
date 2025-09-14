import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings('ignore')


def data_preprocess(df):
    """数据预处理：
        1、读取数据
        2、数据排序
        3、去除重复值
        4、重采样（可选）
        5、缺失值处理
        6、异常值处理
    """
    # ===========读取数据===========
    df = df.sort_values(by='DATATIME', ascending=True)
    print('df.shape:', df.shape)
    print(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")

    # ===========去除重复值===========
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    print('After Dropping duplicates:', df.shape)
    if df['YD15'].isnull().all():
        df['YD15'] = df["ROUND(A.POWER,0)"]

    # 找出离群点并设为空值
    columns = ['ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    for col in columns:
        # 计算四分位距
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # 找出离群点的索引
        outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        outliers_index = df[outliers].index
        # 将离群点设为空值
        df.loc[outliers_index, col] = np.nan

    # ===========重采样（可选） + 线性插值===========
    # 确保 DATETIME 列是 datetime 类型
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])

    # 将 DATETIME 列设置为索引
    df.set_index('DATATIME', inplace=True)

    # 重采样并插值
    # 重采样规则：15分钟 ('15T')，使用右侧标签，闭区间在右侧
    df = df.resample(rule='15T', label='right', closed='right').interpolate(method='linear',
                                                                            limit_direction='both').reset_index()

    # 如果需要进一步处理缺失值，例如使用附近时间点的均值填补
    # 这里可以扩展代码，例如：
    # df['Humidity'] = df['Humidity'].fillna(df['Humidity'].rolling(window=24, center=True).mean())

    # 确保索引是 DatetimeIndex（可选的验证步骤）
    if not isinstance(df.index, pd.DatetimeIndex):
        df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df.set_index('DATATIME', inplace=True)

    # 最终插值（如果需要）
    df = df.interpolate(method='time', limit_direction='both').reset_index()

    columns = ['ROUND(A.POWER,0)', 'YD15']
    for col in columns:
        # 将时间戳转换为日期和时间，并用它们分别进行分组
        if df['DATATIME'].dtype != 'datetime64[ns]':
            df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df['DATE'] = df['DATATIME'].dt.date
        grouped = df.groupby(['DATE'])

        # 统计每个分组内的数据条数
        group_sizes = grouped.size()
        # 打印每个分组的数据条数
        # print(group_sizes)
        # 统计每个分组内YD15的唯一值的数量
        unique_counts = grouped[col].nunique()
        # 找到持续不变的行
        stale_rows = unique_counts[unique_counts == 1].index.tolist()
        stale_indices = []
        for date in stale_rows:
            indices = df.index[(df['DATE'] == date)]
            stale_indices.extend(indices.tolist())

        # 找出YD15持续不变的行设为空值
        # df.drop(stale_indices, inplace=True)
        df.loc[stale_indices, col] = np.nan
        df.drop(columns=['DATE'], inplace=True)

    # 实际风速为负数，实际风速置空
    df.loc[df['ROUND(A.WS,1)'] < 0, 'ROUND(A.WS,1)'] = np.nan
    df.loc[df['YD15'] > 1e10, 'YD15'] = np.nan
    df.loc[df['ROUND(A.POWER,0)'] > 1e10, 'ROUND(A.POWER,0)'] = np.nan
    df.loc[df['YD15'] < -1e10, 'YD15'] = np.nan
    df.loc[df['ROUND(A.POWER,0)'] < -1e10, 'ROUND(A.POWER,0)'] = np.nan
    # 实际风速为0时，功率>0。实际风速>12.5，功率为0
    for col in columns:
        df.loc[(df['ROUND(A.WS,1)'] == 0) & (df[col] > 0), col] = 0
        df.loc[(df['ROUND(A.WS,1)'] > 20) & (df[col] != 0), col] = 0
        df.loc[(df['ROUND(A.WS,1)'] > 5) & (df[col] == 0), col] = np.nan

    print('After Resampling:', df.shape)

    X = df[['WINDSPEED', 'PREPOWER', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']]
    imputer = KNNImputer(n_neighbors=50)
    filled_samples = imputer.fit_transform(X)
    df.loc[X.index, ['WINDSPEED', 'PREPOWER', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']] = filled_samples

    # 将时间列转换为 datetime 类型
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])

    # 将时间列设置为索引
    df.set_index('DATATIME', inplace=True)

    df = df.interpolate(method='linear', limit_direction='both').reset_index()

    return df


def feature_engineer(df):
    """特征工程：时间戳特征"""
    # 时间戳特征
    df['month'] = df.DATATIME.apply(lambda row: row.month, 1)
    df['day'] = df.DATATIME.apply(lambda row: row.day, 1)
    df['hour'] = df.DATATIME.apply(lambda row: row.hour, 1)
    df['minute'] = df.DATATIME.apply(lambda row: row.minute, 1)

    # TODO 挖掘更多特征：差分序列、同时刻风场/邻近风机的特征均值/标准差等

    # # 差分序列
    # df['YD15_diff'] = df['YD15'].diff()
    # df["apow"] = df["ROUND(A.POWER,0)"].diff()
    # df['POWER_diff'] = df['PREPOWER'].diff()
    # df['RWS_diff'] = df['ROUND(A.WS,1)'].diff()
    # df['wp'] = df["WINDSPEED"].diff()
    #
    # # 邻近风机的特征均值/标准差
    # neighboring_features = ['PREPOWER', 'ROUND(A.POWER,0)', 'YD15']
    # df['neighboring_mean'] = df[neighboring_features].mean(axis=1)  # 邻近风机特征均值
    # df['neighboring_std'] = df[neighboring_features].std(axis=1)  # 邻近风机特征标准差
    #
    # df["yd2"] = df["YD15"] * df["YD15"]
    # df["yd3"] = df["YD15"] * df["YD15"] * df["YD15"]
    # df['r12'] = df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"]
    # df['r13'] = df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"] * df["ROUND(A.WS,1)"]
    # df['apow3'] = df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"]
    # df['apow2'] = df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"] * df["ROUND(A.POWER,0)"]
    # df['wp2'] = df["WINDSPEED"] * df["WINDSPEED"]
    # df['wp3'] = df["WINDSPEED"] * df["WINDSPEED"] * df["WINDSPEED"]

    return df
# 'YD15_diff','POWER_diff','RWS_diff','neighboring_mean','neighboring_std','hour_minute_interaction','day_hour_interaction'
