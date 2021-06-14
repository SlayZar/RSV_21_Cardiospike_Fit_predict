import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import base64
import time
from hrvanalysis import remove_outliers, remove_ectopic_beats


# Обнаружение аномалий, накопительное среднее
def anomaly_detected(df):
    new_df = pd.DataFrame()
    for i in df.id.unique():
        new_df_tmp = pd.concat([df[df.id==i].merge(
            pd.DataFrame(remove_outliers(df[df.id==i]['x'], verbose = False), columns=['anomaly_1']),
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
    df['x_cummean'] = df['x'].cumsum() / df['one'].cumsum()
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
    df_train['num_iter'] = df.groupby(['id']).cumcount()+1
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
    counter = 0
    my_bar = st.progress(counter)
    for n in ([1,2,3,5,10,25,50,100,250,500, 10000]):
        counter += (counter + 1) / 11
        counter = min([counter,1])
        my_bar.progress(counter)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n).mean().to_frame(f'mean_rolling_{n}'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n).std().to_frame(f'std_rolling_{n}'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(
            window=n, min_periods=1, center=True).mean().to_frame(f'mean_rolling_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(
            window=n, min_periods=1, center=True).std().to_frame(f'std_rolling_{n}_c'),
               on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .max().to_frame(f'max_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
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
        df_train[f'mean_rolling_1_{n}_rel3'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'mean_rolling_1_{n}_rel4'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'mean_rolling_1_{n}_rel5'] = df_train[f'mean_rolling_1_{n}'] / df_train[f'EMA']
       
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .min().to_frame(f'min_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
        df_train[f'min_rolling_1_{n}_rel1'] = df_train[f'min_rolling_1_{n}'] / df_train[f'mean_value']
        df_train[f'min_rolling_1_{n}_rel2'] = df_train[f'min_rolling_1_{n}'] / df_train[f'max_value']
        df_train[f'min_rolling_1_{n}_rel3'] = df_train[f'min_rolling_1_{n}'] / df_train[f'min_value']
        df_train[f'min_rolling_1_{n}_rel4'] = df_train[f'min_rolling_1_{n}'] / df_train[f'cummean_wo_anomaly']
        df_train[f'min_rolling_1_{n}_rel5'] = df_train[f'min_rolling_1_{n}'] / df_train[f'EMA']
        df_train = df_train.merge(df_train.groupby(['id'])['x'].rolling(window=n, min_periods=1)\
            .median().to_frame(f'median_rolling_1_{n}'),on = np.arange(len(df_train)), how='left').drop(['key_0'], axis=1)
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
    return df_train

@st.cache(suppress_st_warning=True)
def scoring(test_df, path_to_model, new_cols2, tresh, target_col='pred2_bin'):
    test_df = anomaly_detected(test_df)
    test_df = first_prepr(test_df, delete_anomaly = False)
    df_test = test_df[['id', 'time', 'x']].copy()

    test_df = feature_generate(test_df)
    pred_df = test_df[['id', 'time', 'x', 'anomaly__1', 'less066','mean_rolling_new2_1']].copy()

    pred_df = pred_df[(pred_df.anomaly__1==1) |(pred_df.less066==1)|(pred_df.mean_rolling_new2_1==1)]
    pred_df['pred'] = 0
    test_df = test_df[(test_df.anomaly__1==0) &(test_df.less066==0)&(test_df.mean_rolling_new2_1==0)]
    cb = CatBoostClassifier()
    cb.load_model(path_to_model)
    test_df['pred2'] = cb.predict_proba(Pool(test_df[new_cols2]))[:,1]
    test_df[target_col] = (test_df['pred2'] > tresh).astype(int)

    df_test = df_test.merge(pred_df[['id', 'time', 'x', 'pred']], 
                          on =['id', 'time', 'x'], how='left')
    df_test = df_test.merge(test_df[['id', 'time', 'x', 'pred2', 'pred2_bin']], 
                          on =['id', 'time', 'x'], how='left')
    df_test.loc[(df_test.pred2_bin.isnull()), target_col] = df_test.loc[(df_test[target_col].isnull()), 'pred']
    return df_test

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Скачать результаты в формате csv</a>'
    return href

def slider_feats(train, feat, target_col_name):
    try:
        grouped = train.groupby(feat)[target_col_name].mean().to_frame(target_col_name).sort_values(by=target_col_name, ascending=False)
        values = list(grouped.index) # 
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

new_cols2 = ['std_rolling_10_c', 'std_rolling_5_c', 'std_rolling_5', 'std_rolling_100_c',
 'std_value', 'time', 'cummean_diff', 'median_rolling_1_250_rel2',
 'strange_x5', 'std_rolling_25_c', 'std_rolling_3_c', 'min_rolling_1_5_rel3', 'cumsum_wo_anomaly',
 'std_rolling_500_c', 'max_rolling_1_250_rel2', 'mean_rolling_1_50_rel4',
 'median_rolling_1_25_rel4', 'num_iter', 'median_rolling_1_10000',
 'std_rolling_1_25_rel4', 'mean_rolling_1_500_rel1', 'std_rolling_1_25',
 'std_rolling_50_c', 'std_rolling_1_10', 'std_rolling_1_10_rel4',
 'min_rolling_1_10000_rel1', 'max_rolling_1_10000_rel4',
 'std_rolling_1_2_rel4', 'max_value', 'std_rolling_1_5_rel1',
 'std_rolling_1_10000_rel1', 'min_rolling_1_10000', 'median_rolling_1_10000_rel4']
tresh = 0.265
data_path = 'data/test.csv'
target_col_name = 'pred2_bin'

options = st.selectbox('Какие данные скорить?',
         ['Тестовый датасет', 'Загрузить новые данные'], index=1)

if options == 'Тестовый датасет':
    df = pd.read_csv(data_path)
    df.sort_values(by=['id', 'time'], inplace=True)
    res = scoring(df, '1406_best_model', new_cols2, tresh)
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
        res = scoring(df, '1406_best_model', new_cols2, tresh)
        st.markdown('### Скоринг завершён успешно!')

        st.markdown(get_table_download_link(res), unsafe_allow_html=True)
        st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
        
        slider_feats(res, 'id', target_col_name)
        
        
if st.sidebar.button('Анализ важности переменных модели'):
    st.markdown('#### SHAP важности признаков модели')  
    st.image("https://clip2net.com/clip/m392735/16318-clip-315kb.jpg?nocache=1")
    
if st.sidebar.button('Анализ качества модели'):
    st.markdown('#### Точность модели на train-val-test выборках:')  
    st.image("https://clip2net.com/clip/m392735/c6560-clip-81kb.jpg?nocache=1")