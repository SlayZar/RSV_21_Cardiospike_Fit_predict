import streamlit as st
import base64
from hrvanalysis import remove_outliers, remove_ectopic_beats
import pandas as pd
import numpy as np
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features
from catboost import CatBoostClassifier, Pool, CatBoost

# Детекция аномалий
def anomaly_detected(df):
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
  df.drop(['one'], axis=1, inplace=True)
  return df

# Генерация фичей
def features(df):
    df['x-1'] = df['x'].diff(-1)
    df[f'x-1_norm'] = df['x-1'] / df['x']
    df['x-2'] = df['x'].diff(-2)
    df[f'x-2_norm'] = df['x-2'] / df['x']
    df['x-3'] = df['x'].diff(-3)
    df[f'x-3_norm'] = df['x-3'] / df['x']
    df['x-4'] = df['x'].diff(-4)
    df[f'x-4_norm'] = df['x-4'] / df['x']
    df['x-5'] = df['x'].diff(-5)
    df[f'x-5_norm'] = df['x-5'] / df['x']
    df['x-6'] = df['x'].diff(-6)
    df[f'x-6_norm'] = df['x-6'] / df['x']
    df['x+1'] = df['x'].diff(1)
    df[f'x+1_norm'] = df['x+1'] / df['x']
    df['x+2'] = df['x'].diff(2)
    df[f'x+2_norm'] = df['x+2'] / df['x']
    df['x+3'] = df['x'].diff(3)
    df[f'x+3_norm'] = df['x+3'] / df['x']
    df['x+4'] = df['x'].diff(4)
    df[f'x+4_norm'] = df['x+4'] / df['x']
    df['x+5'] = df['x'].diff(5)
    df[f'x+5_norm'] = df['x+5'] / df['x']
    df['x+6'] = df['x'].diff(6)
    df[f'x+5_norm'] = df['x+6'] / df['x']
    df['diff1_6'] = df['x-1'] / df['x-6']
    df['diff+1_6'] = df['x+1'] / df['x+6']
    df[f'x+6_norm'] = df['x'] / df['x+6']
    df['max+'] = df[['x+1', 'x+2', 'x+3', 'x+4', 'x+5', 'x+6']].max(axis=1)
    df['min+'] = df[['x+1', 'x+2', 'x+3', 'x+4', 'x+5', 'x+6']].min(axis=1)
    df['std+'] = df[['x+1', 'x+2', 'x+3', 'x+4', 'x+5', 'x+6']].std(axis=1)
    df['max-'] = df[['x-1', 'x-2', 'x-3', 'x-4', 'x-5', 'x-6']].max(axis=1)
    df['min-'] = df[['x-1', 'x-2', 'x-3', 'x-4', 'x-5', 'x-6']].min(axis=1)
    df['std-'] = df[['x-1', 'x-2', 'x-3', 'x-4', 'x-5', 'x-6']].std(axis=1)
    df['max+-'] = df[['x-1', 'x-2', 'x-3', 'x+1', 'x+2', 'x+3']].max(axis=1)
    df['min+-'] = df[['x-1', 'x-2', 'x-3', 'x+1', 'x+2', 'x+3']].min(axis=1)
    df['std+-'] = df[['x-1', 'x-2', 'x-3', 'x+1', 'x+2', 'x+3']].std(axis=1)
    df['norm_x'] = 60000 / df['x']
    df = df.merge(df.groupby(['id'])['norm_x'].rolling(
            window=10, min_periods=1, center=True).std().to_frame(f'norm_std_rolling_10_c'),
               on = np.arange(len(df)), how='left').drop(['key_0'], axis=1)
    return df


@st.cache(suppress_st_warning=True)
def scoring(test_df, path_to_model, new_cols2, tresh, target_col='pred2_bin'):
    df_test = test_df[['id', 'time', 'x']].copy()
    test_df = anomaly_detected(test_df)
    test_df = test_df[test_df['anomaly__1'] == 0]
    data2 = pd.DataFrame()
    for ids in list(test_df.id.unique()):
        df = test_df[test_df.id == ids]
        data2 = data2.append(features(df.copy()))
    cb = CatBoostClassifier()
    cb.load_model(path_to_model)
    test_df['pred2'] = cb.predict_proba(Pool(data2[new_cols2]))[:,1]
    test_df[target_col] = (test_df['pred2'] > tresh).astype(int)
    df_test = df_test.merge(test_df[['id', 'time', 'x', 'pred2', target_col]], 
                          on =['id', 'time', 'x'], how='left')
    df_test.loc[(df_test[target_col].isnull()), target_col] = 0
    return df_test

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="Cardio_scorng_model_results.csv">Скачать результаты в формате csv</a>'
    return href

def slider_feats(train, feat, target_col_name):
    try:
        grouped = train.groupby(feat)[target_col_name].mean().to_frame(target_col_name).sort_values(by=target_col_name, ascending=False)
        values = list(grouped.index) 
    except:
        values = sorted(train[feat].unique())
    
    st.write('Результат работы модели для различных ID')
    ids = st.selectbox(f'Выберите {feat} (сортировка в порядке уменьшения числа выявленных аномалий)', values)
    df_samp = train[train['id']==ids].copy()
    df_samp.set_index('time', inplace=True)
    df_samp['Аномалия ритма сердца'] = df_samp['x'] * df_samp[target_col_name].replace(0, np.nan)
    try:
        st.line_chart(df_samp[['x', 'Аномалия ритма сердца']])
    except:
        pass

st.set_page_config("Fit_Prdeict Cardiospike demo")
st.image("https://i.ibb.co/Vwhhs7J/image.png", width=150)

if st.sidebar.button('Очистить кэш'):
    st.caching.clear_cache()

new_cols2 = ['time', 'x+1', 'x-1', 'x-2', 'x+4', 'x-3', 'x+2', 'x-4', 'x+3', 'x-5', 'x+5', 'x+6',  
   'x', 'x-6', 'max+',  'min+', 'std+', 'max-',  'min-',  'std-', 'max+-',  'min+-',   'std+-']
tresh = 0.4824120603015075
data_path = 'data/test.csv'
target_col_name = 'prediction'

st.markdown('## Детектор ковидных аномалий на ритмограмме')

options = st.selectbox('Какие данные скорить?',
         ['Тестовый датасет', 'Загрузить новые данные'], index=1)

if options == 'Тестовый датасет':
    df = pd.read_csv('data/test.csv')
    df.sort_values(by=['id', 'time'], inplace=True)
    res = scoring(df, 'models/best_model', new_cols2, tresh, target_col = target_col_name)
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
        res = scoring(df, 'models/best_model', new_cols2, tresh, target_col = target_col_name)
        st.markdown('### Скоринг завершён успешно!')

        st.markdown(get_table_download_link(res), unsafe_allow_html=True)
		
	
        st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
        
        slider_feats(res, 'id', target_col_name)
        
        
if st.sidebar.button('Анализ важности переменных модели'):
    st.markdown('#### SHAP важности признаков модели')  
    st.image("https://clip2net.com/clip/m392735/11832-clip-461kb.jpg?nocache=1")
    
if st.sidebar.button('Анализ качества модели'):
    st.markdown('#### Точность модели на train-val-test выборках:')  
    st.image("https://clip2net.com/clip/m392735/067c5-clip-125kb.png?nocache=1")
    st.image("https://clip2net.com/clip/m392735/c50c0-clip-60kb.png?nocache=1")
