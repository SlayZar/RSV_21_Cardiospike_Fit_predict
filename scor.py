import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import base64
import time
from hrvanalysis import remove_outliers, remove_ectopic_beats
from hrvanalysis import get_sampen, get_time_domain_features, get_geometrical_features, get_frequency_domain_features,\
 get_csi_cvi_features, get_poincare_plot_features
 
# Детектирование пиков
def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    if qrs_filter is None:
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)
    return ecg_signal[similarity > threshold].index, similarity

# Обнаружение аномалий, накопительное среднее
def anomaly_detected(df):
  df['peak'] = 0
  res = pd.DataFrame()
  for i in df.id.unique():
    df.loc[detect_peaks(df[df.id==i]['x'], threshold=0.3)[0], 'peak']=1
    df.loc[(df.id==i),'peak2'] = detect_peaks(df[df.id==i]['x'], threshold=0.3)[1]
  new_df = pd.DataFrame()
  for i in df.id.unique():
    new_df_tmp = pd.concat([df[df.id==i].merge(
        pd.DataFrame(remove_outliers(df[df.id==i]['x']), columns=['anomaly_1']),
                    on = np.arange(len(df[df.id==i]))).drop(['key_0'], axis=1)])
    new_df = pd.concat([new_df, new_df_tmp])
  new_df2 = pd.DataFrame()
  for i in df.id.unique():
    new_df2_tmp = pd.concat([df[df.id==i].merge(
        pd.DataFrame(remove_ectopic_beats(list(df[df.id==i]['x'])), columns=['anomaly_2']),
                    on = np.arange(len(df[df.id==i]))).drop(['key_0'], axis=1)])
    new_df2 = pd.concat([new_df2, new_df2_tmp])
  new_df['anomaly__1'] = (new_df['anomaly_1'].isnull()).astype(int)
  new_df2['anomaly__2'] = (new_df2['anomaly_2'].isnull()).astype(int)
  new_df.drop(['anomaly_1'], axis=1, inplace=True)
  new_df2.drop(['anomaly_2'], axis=1, inplace=True)
  df = new_df.merge(new_df2[['id', 'time', 'anomaly__2']], on = ['id', 'time'])
  df['EMA'] = df['x'].ewm(span=40,adjust=False).mean()
  df['one'] = 1
  df['x_cummean'] = df.groupby(['id'])['x'].cumsum() / df.groupby(['id'])['one'].cumsum()
  df['peak_cummean'] = df.groupby(['id'])['peak'].cumsum() / df.groupby(['id'])['one'].cumsum()
  df['is_anomaly'] = df['anomaly__1'] + df['anomaly__2']
  df.drop(['one'], axis=1, inplace=True)
  return df

# Накопительное среднее без аномалий, фичи по количеству аномалий
def first_prepr(df_train, delete_anomaly = True):
    df_train.sort_values(by=['id', 'time'], inplace=True)
    df_train['anomaly1_cumsum'] = df_train.groupby(['id'])['anomaly__1'].cumsum()
    df_train['anomaly2_cumsum'] = df_train.groupby(['id'])['anomaly__2'].cumsum()
    df_train['one'] = 1
    # Номер итерации
    df_train['num_iter'] = df_train.groupby(['id']).cumcount()+1    
    df_train['norm_x'] = 60000 / df_train['x']
    df_train = df_train.merge(df_train.groupby(['id'])['norm_x'].mean().to_frame('norm_mean_value'),
                                                      left_on = ['id'], right_index=True)
    df_train = df_train.merge(df_train.groupby(['id'])['x'].mean().to_frame('mean_value'),
                                                      left_on = ['id'], right_index=True)\
        .merge(df_train.groupby(['id'])['x'].std().to_frame('std_value'), left_on = ['id'], right_index=True)\
            .merge(df_train.groupby(['id'])['x'].min().to_frame('min_value'), left_on = ['id'], right_index=True)\
                    .merge(df_train.groupby(['id'])['x'].max().to_frame('max_value'), left_on = ['id'], right_index=True)\
              .merge(df_train[df_train.is_anomaly==0].groupby(['id'])['x'].mean().to_frame('mean_not_anomaly'), 
                     left_on=['id'], right_index=True, how='left').\
              merge(df_train[df_train.is_anomaly==0].groupby(['id'])['x'].cumsum().to_frame('cumsum_wo_anomaly'), 
                     left_index =True, right_index=True, how='left').\
              merge(df_train[df_train.is_anomaly==0].groupby(['id'])['one'].cumsum().to_frame('cumcount_wo_anomaly'), 
                     left_index =True, right_index=True, how='left')
    df_train['cummean_wo_anomaly'] = df_train['cumsum_wo_anomaly'] / df_train['cumcount_wo_anomaly']
    for i in range(100):
      if df_train.cummean_wo_anomaly.isnull().sum()>0:
        df_train['cummean_wo_anomaly_sh_1'] = df_train['cummean_wo_anomaly'].shift(1)
        df_train.loc[df_train.cummean_wo_anomaly.isnull(), 'cummean_wo_anomaly'] = \
              df_train.loc[df_train.cummean_wo_anomaly.isnull(), 'cummean_wo_anomaly_sh_1']
    df_train['cummean_diff'] = df_train['cummean_wo_anomaly'] / df_train['x_cummean']
    df_train['not_anomaly_perc'] = df_train['cumcount_wo_anomaly'] / df_train['num_iter']
    df_train.drop(['cummean_wo_anomaly_sh_1', 'one', 'cumcount_wo_anomaly'], axis=1, inplace=True)
    
    if delete_anomaly:
      df_train = df_train[df_train.anomaly__1==0]
    return df_train

# Куча фичей
def feature_generate(df_train):
    df_train['peak_new'] = 0
    res = pd.DataFrame()
    for i in df_train.id.unique():
        df_train.loc[detect_peaks(df_train[df_train.id==i]['x'], threshold=0.3)[0], 'peak_new']=1
        df_train.loc[(df_train.id==i),'peak_new2'] = detect_peaks(df_train[df_train.id==i]['x'], threshold=0.3)[1]
    counter = 0
    my_bar = st.progress(counter)
    for n in ([1,2,3,5,10,25,50,100,250,500, 10000]):
        counter += (counter + 1) / 11
        counter = min([counter,1])
        my_bar.progress(counter)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n).mean().to_frame(f'mean_rolling_{n}'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['norm_x'].rolling(window=n, min_periods=1).mean()\
                                  .to_frame(f'norm_mean_rolling_1_{n}'),
                                  on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n).std().to_frame(f'std_rolling_{n}'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['norm_x'].rolling(window=n, min_periods=1)\
                                    .std().to_frame(f'norm_std_rolling_1_{n}'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=n)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(
            window=indexer, min_periods=1).mean().to_frame(f'mean_rolling_1_{n}_f'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        if n >=100:
          df_train = df_train.merge(df_train.groupby(['id'])['peak'].rolling(window=n, min_periods=1, center=True).sum()\
                                  .to_frame(f'peak_sum_c_1_rolling_{n}'),
                                  on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
          df_train = df_train.merge(df_train.groupby(['id'])['peak'].rolling(window=n, min_periods=1, center=True).mean()\
                                  .to_frame(f'peak_mean_c_1_rolling_{n}'),
                                  on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
          df_train = df_train.merge(df_train.groupby(['id'])['peak_new'].rolling(window=n, min_periods=1, center=True).sum()\
                                  .to_frame(f'peak_new_sum_c_1_rolling_{n}'),
                                  on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
          df_train = df_train.merge(df_train.groupby(['id'])['peak_new'].rolling(window=n, min_periods=1, center=True).mean()\
                                  .to_frame(f'peak_new_mean_c_1_rolling_{n}'),
                                  on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)

        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(
            window=n, min_periods=1, center=True).mean().to_frame(f'mean_rolling_1_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['norm_x'].rolling(
            window=n, min_periods=1, center=True).mean().to_frame(f'norm_mean_rolling_1_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(
            window=n, min_periods=1, center=True).std().to_frame(f'std_rolling_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['norm_x'].rolling(
            window=n, min_periods=1, center=True).std().to_frame(f'norm_std_rolling_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .max().to_frame(f'max_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1, center=True)\
            .max().to_frame(f'max_rolling_1_{n}_c'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1, center=True)\
            .min().to_frame(f'min_rolling_1_{n}_c'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train[f'max_rolling_1_{n}_rel1'] = df_train[f'max_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'max_rolling_1_{n}_rel2'] = df_train[f'max_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'max_rolling_1_{n}_rel3'] = df_train[f'max_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'max_rolling_1_{n}_rel4'] = df_train[f'max_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'max_rolling_1_{n}_rel5'] = df_train[f'max_rolling_1_{n}'] / df_train[f'EMA']
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .mean().to_frame(f'mean_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['anomaly__1'].rolling(window=n, min_periods=1)\
            .mean().to_frame(f'mean_rolling_new_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['anomaly__2'].rolling(window=n, min_periods=1)\
            .mean().to_frame(f'mean_rolling_new2_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train[f'mean_rolling_1_{n}_rel1'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'mean_rolling_1_{n}_rel2'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'mean_rolling_{n}_rel1_c'] = df_train[f'mean_rolling_1_{n}_c'] / df_train[f'mean_value']
        df_train[f'mean_rolling_{n}_rel2_c'] = df_train[f'mean_rolling_1_{n}_c'] / df_train[f'max_value']
        df_train[f'mean_rolling_{n}_rel1_f'] = df_train[f'mean_rolling_1_{n}_f'] / df_train[f'mean_value']
        df_train[f'mean_rolling_{n}_rel2_f'] = df_train[f'mean_rolling_1_{n}_f'] / df_train[f'max_value']
        df_train[f'mean_rolling_1_{n}_rel3'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'mean_rolling_1_{n}_rel4'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'mean_rolling_1_{n}_rel5'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'EMA']
        df_train[f'norm_mean_rolling_1_{n}_rel1'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'norm_mean_rolling_1_{n}_rel2'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'norm_mean_rolling_1_{n}_rel3'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'norm_mean_rolling_1_{n}_rel4'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'norm_mean_rolling_1_{n}_rel5'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'EMA']
        df_train[f'norm_mean_rolling_1_{n}_rel5'] = df_train[f'norm_mean_rolling_1_{n}'] / df_train[f'norm_mean_value']
       
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .min().to_frame(f'min_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train[f'min_rolling_1_{n}_rel1'] = df_train[f'min_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'min_rolling_1_{n}_rel2'] = df_train[f'min_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'min_rolling_1_{n}_rel3'] = df_train[f'min_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'min_rolling_1_{n}_rel4'] = df_train[f'min_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'min_rolling_1_{n}_rel5'] = df_train[f'min_rolling_1_{n}'] / df_train[f'EMA']
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .median().to_frame(f'median_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1, center=True)\
            .median().to_frame(f'median_rolling_1_{n}_c'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train[f'median_rolling_1_{n}_rel1'] = df_train[f'median_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'median_rolling_1_{n}_rel2'] = df_train[f'median_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'median_rolling_1_{n}_rel3'] = df_train[f'median_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'median_rolling_1_{n}_rel4'] = df_train[f'median_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'median_rolling_1_{n}_rel5'] = df_train[f'median_rolling_1_{n}'] / df_train[f'EMA']
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .std().to_frame(f'std_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)        
        df_train[f'std_rolling_1_{n}_rel1'] = df_train[f'std_rolling_1_{n}'] / df_train[f'std_value']
        df_train[f'std_rolling_1_{n}_rel2'] = df_train[f'std_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'std_rolling_1_{n}_rel3'] = df_train[f'std_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'std_rolling_1_{n}_rel4'] = df_train[f'std_rolling_1_{n}'] / df_train[f'EMA']       
        df_train[f'norm_std_rolling_1_{n}_rel1'] = df_train[f'norm_std_rolling_1_{n}'] / df_train[f'std_value']
        df_train[f'norm_std_rolling_1_{n}_rel2'] = df_train[f'norm_std_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'norm_std_rolling_1_{n}_rel3'] = df_train[f'norm_std_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'norm_std_rolling_1_{n}_rel4'] = df_train[f'norm_std_rolling_1_{n}'] / df_train[f'EMA']

        df_train[f'up_anomaly_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel3'] + \
                                        df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'up_anomaly2_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel3'] + \
                                        2* df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'up_anomaly3_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel3'] + \
                                        3 * df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'down_anomaly_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel3'] - \
                                        df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'down_anomaly2_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel3'] - \
                                        2*df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'down_anomaly3_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel3'] - \
                                       3* df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'is_anomaly_{n}'] = df_train[f'up_anomaly_{n}'] + df_train[f'up_anomaly2_{n}'] + \
          df_train[f'up_anomaly3_{n}'] + df_train[f'down_anomaly_{n}'] + df_train[f'down_anomaly2_{n}'] + \
            df_train[f'down_anomaly3_{n}']

        df_train[f'4up_anomaly_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel4'] + \
                                        df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4up_anomaly2_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel4'] + \
                                        2* df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4up_anomaly3_{n}'] = (df_train['x'] > df_train[f'mean_rolling_1_{n}_rel4'] + \
                                        3 * df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4down_anomaly_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel4'] - \
                                        df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4down_anomaly2_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel4'] - \
                                        2*df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4down_anomaly3_{n}'] = (df_train['x'] < df_train[f'mean_rolling_1_{n}_rel4'] - \
                                       3* df_train[f'std_rolling_1_{n}']).astype(int)
        df_train[f'4is_anomaly_{n}'] = df_train[f'4up_anomaly_{n}'] + df_train[f'4up_anomaly2_{n}'] + \
          df_train[f'4up_anomaly3_{n}'] + df_train[f'4down_anomaly_{n}'] + df_train[f'4down_anomaly2_{n}'] + \
            df_train[f'4down_anomaly3_{n}']        
    df_train['strange_x1'] = df_train['x'] / df_train[f'mean_value']
    df_train['strange_x2'] = df_train['x'] / df_train[f'cummean_wo_anomaly']
    df_train['strange_x3'] = df_train['x'] / df_train[f'x_cummean']
    df_train['strange_x4'] = df_train['x'] / df_train[f'EMA']
    df_train['strange_x5'] = df_train['x'] / df_train[f'mean_rolling_5']
    df_train['strange_x6'] = df_train['x'] / df_train[f'mean_rolling_100']
    df_train.num_iter = (df_train.num_iter//100)
    df_train['less066'] = (df_train['x'] / df_train['cummean_wo_anomaly'] <0.66).astype(int)
    df_train['more133'] = (df_train['x'] / df_train['cummean_wo_anomaly'] >1.33).astype(int)

    df_train = df_train.merge(
    (df_train.groupby(['id'])['time'].max() / 60000).astype(int).to_frame('mins'),
    left_on = 'id', right_index=True)
    
    df_train = df_train.merge(
    (df_train.groupby(['id'])['time'].max()).astype(int).to_frame('max_time'),
    left_on = 'id', right_index=True)
    df_train['time_share'] = df_train['time'] / df_train['max_time']

    df_train = df_train.merge(
        (df_train.groupby(['id'])['time'].max() / 1000).astype(int).to_frame('secs'),
        left_on = 'id', right_index=True)

    df_train['anomaly_feat'] = df_train['max_rolling_1_10000'] / df_train['median_rolling_1_10000']
    df_train['anomaly_feat3'] = df_train['max_rolling_1_3'] / df_train['median_rolling_1_10']
    df_train['std_diff'] = df_train['std_rolling_10_c'] / df_train['norm_std_rolling_10_c']

    df_train['my_feat1'] = abs(df_train['max_rolling_1_10'] / df_train[f'median_rolling_1_250']) + abs(df_train['min_rolling_1_10'] / df_train[f'median_rolling_1_250'])
    df_train['my_feat2'] = abs(df_train['max_rolling_1_3'] / df_train[f'median_rolling_1_250']) +  abs(df_train['min_rolling_1_3'] / df_train[f'median_rolling_1_250'])
    df_train['my_feat3'] = abs(df_train['max_rolling_1_10_c'] / df_train[f'median_rolling_1_250']) + abs(df_train['min_rolling_1_10_c'] / df_train[f'median_rolling_1_250'])
    df_train['my_feat4'] = abs(df_train['max_rolling_1_10_c'] / df_train[f'median_rolling_1_250_c']) + abs(df_train['min_rolling_1_10_c'] / df_train[f'median_rolling_1_250_c'])
    return df_train

def new_feats(df_train):
  get_sampen_df_res = pd.DataFrame()
  get_time_domain_features_res = pd.DataFrame()
  get_geometrical_features_res = pd.DataFrame()
  get_frequency_domain_features_res = pd.DataFrame()
  get_csi_cvi_features_res = pd.DataFrame()
  get_poincare_plot_features_res = pd.DataFrame()
  for i in df_train.id.unique():
    get_sampen_df =  pd.DataFrame(get_sampen(df_train[df_train.id==i]['x']), index=[i])
    get_sampen_df['id'] = i 
    get_sampen_df_res = pd.concat([get_sampen_df, get_sampen_df_res])
    get_time_domain_features_df =  pd.DataFrame(get_time_domain_features(df_train[df_train.id==i]['x']), index=[i])
    get_time_domain_features_df['id'] = i 
    get_time_domain_features_res = pd.concat([get_time_domain_features_df, get_time_domain_features_res])
    get_geometrical_features_df =  pd.DataFrame(get_geometrical_features(df_train[df_train.id==i]['x']), index=[i])
    get_geometrical_features_df['id'] = i 
    get_geometrical_features_res = pd.concat([get_geometrical_features_df, get_geometrical_features_res])
    get_frequency_domain_features_df =  pd.DataFrame(get_frequency_domain_features(df_train[df_train.id==i]['x']), index=[i])
    get_frequency_domain_features_df['id'] = i 
    get_frequency_domain_features_res = pd.concat([get_frequency_domain_features_df, get_frequency_domain_features_res])
    get_csi_cvi_features_df =  pd.DataFrame(get_csi_cvi_features(df_train[df_train.id==i]['x']), index=[i])
    get_csi_cvi_features_df['id'] = i 
    get_csi_cvi_features_res = pd.concat([get_csi_cvi_features_df, get_csi_cvi_features_res])
    get_poincare_plot_features_df =  pd.DataFrame(get_poincare_plot_features(df_train[df_train.id==i]['x']), index=[i])
    get_poincare_plot_features_df['id'] = i 
    get_poincare_plot_features_res = pd.concat([get_poincare_plot_features_df, get_poincare_plot_features_res])
  df_train = df_train.merge(get_sampen_df_res, on=['id'])
  df_train = df_train.merge(get_time_domain_features_res, on=['id'])
  df_train = df_train.merge(get_geometrical_features_res, on=['id'])
  df_train = df_train.merge(get_frequency_domain_features_res, on=['id'])
  df_train = df_train.merge(get_csi_cvi_features_res, on=['id'])
  df_train = df_train.merge(get_poincare_plot_features_res, on=['id'])
  return df_train

@st.cache(suppress_st_warning=True)
def scoring(test_df, path_to_model, new_cols2, tresh, target_col='pred2_bin'):
    df_test = test_df[['id', 'time', 'x']].copy()
    test_df = anomaly_detected(test_df)
    test_df = first_prepr(test_df)
    test_df = test_df[test_df.x<1100]

    test_df = feature_generate(test_df)
    test_df = new_feats(test_df)
    test_df['change'] = test_df['x'] / test_df['median_rolling_1_100']
    test_df = test_df[test_df['change']>0.8]
    cb = CatBoostClassifier()
    cb.load_model(path_to_model)
    test_df['pred2'] = cb.predict_proba(Pool(test_df[new_cols2]))[:,1]
    st.write('Sc')
    test_df[target_col] = (test_df['pred2'] > tresh).astype(int)
    df_test = df_test.merge(test_df[['id', 'time', 'x', 'pred2', target_col]], 
                          on =['id', 'time', 'x'], how='left')
    df_test.loc[(df_test[target_col].isnull()), target_col] = 0
    return df_test

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Скачать результаты в формате csv</a>'
    return href

def slider_feats(train, feat, target_col_name):
    try:
        grouped = train.groupby(feat)[target_col_name].mean().to_frame(target_col_name).sort_values(by=target_col_name, ascending=False)
        values = list(grouped.index) 
    except:
        values = sorted(train[feat].unique())
    
    st.write('Результат работы модели для различных ID')
    ids = st.selectbox(f'Выберите {feat}', values)
    df_samp = train[train['id']==ids].copy()
    df_samp.set_index('time', inplace=True)
    df_samp['Аномалия ритма сердца'] = df_samp['x'] * df_samp[target_col_name].replace(0, np.nan)
    try:
        st.line_chart(df_samp[['x', 'Аномалия ритма сердца']])
    except:
        pass

if st.sidebar.button('Очистить кэш'):
    st.caching.clear_cache()

new_cols2 = ['std_rolling_10_c', 'norm_std_rolling_10_c',
 'std_rolling_5_c', 'nni_20', 'max_time',
 'secs', 'std_rolling_5', 'median_rolling_1_100_rel5',
 'strange_x5', 'median_rolling_1_25_rel4', 'anomaly_feat3', 'norm_std_rolling_5_c',
 'std_rolling_25_c', 'std_rolling_3_c', 'mins', 'median_rolling_1_500_rel1',
 'std_rolling_1_5_rel2', 'my_feat4', 'norm_std_rolling_25_c',
 'peak2', 'peak_mean_c_1_rolling_500', 'peak_sum_c_1_rolling_500',
 'median_rolling_1_10000_rel4', 'triangular_index', 'std_rolling_50_c', 'std_rolling_1_10',
 'time', 'std_rolling_2_c', 'norm_std_rolling_50_c', 'std_value', 'median_rolling_1_50_rel5',
 'std_rolling_1_25_rel3', 'mean_rolling_1_50_f', 'std_rolling_1_10_rel3',
 'min_rolling_1_10000', 'min_rolling_1_5_rel3', 'peak_new2',
 'time_share', 'norm_mean_rolling_1_25_rel4',
 'median_rolling_1_25_rel5', 'std_rolling_1_50_rel3',
 'anomaly2_cumsum', 'max_rolling_1_50_c', 'std_rolling_1_25']
tresh = 0.204
data_path = 'data/test.csv'
target_col_name = 'prediction'

options = st.selectbox('Какие данные скорить?',
         ['Тестовый датасет', 'Загрузить новые данные'], index=1)

if options == 'Тестовый датасет':
    df = pd.read_csv('data/test.csv')
    df.sort_values(by=['id', 'time'], inplace=True)
    res = scoring(df, 'models/1906_best_model', new_cols2, tresh, target_col = target_col_name)
    st.markdown('### Скоринг завершён успешно!')

    st.markdown(get_table_download_link(res), unsafe_allow_html=True)
    st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
    slider_feats(res, 'id', target_col_name)
else:
    file_buffer = st.file_uploader(label = 'Выберите датасет')
    if file_buffer:
        try:
            df = pd.read_csv(file_buffer, encoding=None)
        except:
            st.write('Файл некорректен!')
        assert df.shape[1] == 3 or df.shape[1] == 4
        st.markdown('#### Файл корректный!')  
        st.write('Пример данных из файла:')
        st.dataframe(df.sample(3))  
        res = scoring(df, 'models/1906_best_model', new_cols2, tresh, target_col = target_col_name)
        st.markdown('### Скоринг завершён успешно!')

        st.markdown(get_table_download_link(res), unsafe_allow_html=True)
        st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
        
        slider_feats(res, 'id', target_col_name)
        
        
if st.sidebar.button('Анализ важности переменных модели'):
    st.markdown('#### SHAP важности признаков модели')  
    st.image("https://clip2net.com/clip/m392735/06587-clip-294kb.jpg?nocache=1")
    
if st.sidebar.button('Анализ качества модели'):
    st.markdown('#### Точность модели на train-val-test выборках:')  
    st.image("https://clip2net.com/clip/m392735/2d2ea-clip-62kb.jpg?nocache=1")
