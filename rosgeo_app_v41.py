import streamlit as st
import tensorflow
import openpyxl
import requests
import urllib.request
import tempfile
import shutil
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np

# делаем сайдбар с выбором моделей
st.sidebar.header('ВЫБОР МОДЕЛЕЙ ДЛЯ ПРЕДСКАЗАНИЯ ГЕОДАННЫХ')

# выбор моделей для предсказания коллекторов делаем чекбоксами, так как будет реализована логика предсказания коллектора
# несколькими моделями, а затем выбор наиблее часто предсказанного как верного
st.sidebar.subheader('Предсказание типа коллектора')
# bagurin_cb = st.sidebar.checkbox('Багурин М.')
# grigorevckiy_cb = st.sidebar.checkbox('Григоревский К.')
baseline_cb = st.sidebar.checkbox('Базовая модель')
soldatov_cb = st.sidebar.checkbox('Солдатов А.', value=True)
st.sidebar.write('---')

# выбор моделей KNEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Предсказание KNEF')
knef_radio = st.sidebar.checkbox('Новиков А. (ilro)', value=True)
# knef_radio = st.sidebar.radio('Модели KNEF', ('Новиков А. (ilro)'))
st.sidebar.write('---')

# выбор моделей KPEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Предсказание KPEF')
kpef_radio = st.sidebar.checkbox('Фадеев Ю.', value=True)
# kpef_radio = st.sidebar.radio('Модели KPEF', ('Фадеев Ю.'))

# основной блок с выводом информации
st.title('ПРЕДСКАЗАНИЕ ГЕОДАННЫХ')
st.write('---')

st.header('Ввод данных для обработки')

# вызываем блок для загрузки файла
uploaded_file = st.file_uploader(label='Выберите файл в формате xls для обработки')

# используем разное количество заголовков колонок, так как какие-то модели работают с 8 столбцами, какие-то с 9-10
cols_collectors = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
cols_KNEF = ['ГЛУБИНА', 'GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']

# осуществляем предобработку данных, загруженных через файл
if uploaded_file is not None:
        st.write('*Файл загружен успешно*')

        df = pd.read_excel(uploaded_file, engine='openpyxl', header=1)

        df = df.dropna(axis='index', how='any')
        df = df.drop([0])
        df.reset_index(drop=True, inplace=True)
        for col in df.columns:
            if col not in cols_KNEF:
                df.pop(col)
        df = df.reindex(columns=cols_KNEF)

        def get_x_data(dataframe, cols_collectors=cols_collectors, cols_KNEF=cols_KNEF):
            get_x_collectors = dataframe[cols_collectors].values.astype(np.float32)
            get_x_KNEF = dataframe[cols_KNEF].values.astype(np.float32)
            print(f'Размер get_x_collectors: {get_x_collectors.shape}. Размер get_x_KNEF: {get_x_KNEF.shape}.')
            return get_x_collectors, get_x_KNEF

        predict_collectors, predict_KNEF = get_x_data(df)

# осуществляем предобработку данных, загруженных вручную
else:
    def accept_user_data():
        st.write('')
        st.write('**или введите данные вручную**')
        col1, col2, col3 = st.columns(3, gap='medium')

        with col1:
            depth_korr = st.number_input('ГЛУБИНА')
            ggkp_korr = st.number_input('GGKP_korr')
            gk_korr = st.number_input('GK_korr')

        with col2:
            pe_korr = st.number_input('PE_korr')
            ds_korr = st.number_input('DS_korr')
            dtp_korr = st.number_input('DTP_korr')

        with col3:
            wi_korr = st.number_input('Wi_korr')
            bk_korr = st.number_input('BK_korr')
            bmk_korr = st.number_input('BMK_korr')

        data = {'ГЛУБИНА': depth_korr,
                'GGKP_korr': ggkp_korr,
                'GK_korr': gk_korr,
                'PE_korr': pe_korr,
                'DS_korr': ds_korr,
                'DTP_korr': dtp_korr,
                'Wi_korr': wi_korr,
                'BK_korr': bk_korr,
                'BMK_korr': bmk_korr
                }
        user_prediction_data = pd.DataFrame(data, index=[0])

        return user_prediction_data

    predict_KNEF = accept_user_data()
    predict_collectors = predict_KNEF.drop(columns=['ГЛУБИНА'],axis=1)

st.write('---')
st.subheader('Введенные данные')


def args_to_types(output):
    res = []
    for i in output:
        if i == 0:
            res.append('2')
        elif i == 1:
            res.append('4')
        elif i == 2:
            res.append('80')
        else:
            res.append('0')
    res = pd.DataFrame(res)
    return res


if uploaded_file is not None:
    st.write(df)
else:
    st.write('*Загрузите файл или введите данные вручную*')
    st.write(predict_KNEF)

@st.experimental_singleton
def load_models():

    # МОДЕЛИ РАСПОЗНАВАНИЯ КОЛЛЕКТОРОВ
    # модель Александра Солдатова
    with urllib.request.urlopen('http://ilro.ru/Collectors/Soldatov/model_Soldatov_RF_Collectors_model_small.pkl') as url_soldatov:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_soldatov:
            shutil.copyfileobj(url_soldatov, tmp_soldatov)

    with open(tmp_soldatov.name, 'rb') as f:
        loaded_model_soldatov_collectors = pickle.load(f)


    # МОДЕЛИ РАСПОЗНАВАНИЯ KNEF
    # модель Алексея Новикова
    # with urllib.request.urlopen('http://ilro.ru/KNEF/Novikov/model_ilro_KNEF_model.json') as url_novikov_model:
    #     with tempfile.NamedTemporaryFile(delete=False) as tmp_novikov_model:
    #         shutil.copyfileobj(url_novikov_model, tmp_novikov_model)
    #
    # with urllib.request.urlopen('http://ilro.ru/KNEF/Novikov/model_ilro_KNEF_weights.h5') as url_novikov_weights:
    #     with tempfile.NamedTemporaryFile(delete=False) as tmp_novikov_weights:
    #         shutil.copyfileobj(url_novikov_weights, tmp_novikov_weights)
    #
    # with open(tmp_novikov_model.name, 'rb') as json_file_KNEF:
    #     # loaded_model_json_KNEF = json_file_KNEF.read()
    #     # json_file_KNEF.close()
    #     loaded_model_KNEF = model_from_json(json_file_KNEF)
    #     json_file_KNEF.close()
    #
    # with open(tmp_novikov_weights.name, 'rb') as tmp_novikov_weights:
    #     loaded_model_KNEF.load_weights(tmp_novikov_weights)


    json_file_collectors = open('Models/COLLECTORS/Collectors_base_model.json', 'r')
    loaded_model_json_collectors = json_file_collectors.read()
    json_file_collectors.close()
    loaded_model_collectors = model_from_json(loaded_model_json_collectors)
    loaded_model_collectors.load_weights('Models/COLLECTORS/Collectors_base_model.h5')
    print('Loaded model COLLECTORS from disk')


    json_file_KNEF = open('Models/KNEF/model_ilro_KNEF_model.json', 'r')
    loaded_model_json_KNEF = json_file_KNEF.read()
    json_file_KNEF.close()
    loaded_model_KNEF = model_from_json(loaded_model_json_KNEF)
    loaded_model_KNEF.load_weights('Models/KNEF/model_ilro_KNEF_weights.h5')
    print('Loaded model KNEF from disk')

    json_file_KPEF = open('Models/KPEF/KPEF_baseline_model.json', 'r')
    loaded_model_json_KPEF = json_file_KPEF.read()
    json_file_KPEF.close()
    loaded_model_KPEF = model_from_json(loaded_model_json_KPEF)
    loaded_model_KPEF.load_weights('Models/KPEF/KPEF_baseline_weights.h5')
    print('Loaded model KPEF from disk')

    return loaded_model_soldatov_collectors, loaded_model_collectors, loaded_model_KNEF, loaded_model_KPEF

loaded_model_soldatov_collectors, loaded_model_collectors, loaded_model_KNEF, loaded_model_KPEF = load_models()

result = st.button('Классифицировать')

# функция предсказания типа коллектора на основе выбранной модели и исходных данных
def preds_argmax_collectors(model='', x_test=''):

    if len(x_test)>1:
        preds_collectors = model.predict(x_test)
        pred_args_collector = np.argmax(preds_collectors, axis=1)
        out_collectors = args_to_types(pred_args_collector)

    else:
        preds_collectors = model.predict(x_test)
        pred_args_collector = np.argmax(preds_collectors, axis=1)
        out_collectors = args_to_types(pred_args_collector)
        # out_collectors = f'Коллектор: {out_collectors[0][0]}'
        out_collectors = out_collectors[0][0]

    return out_collectors

# функция прогноза KNEF
def preds_KNEF(model='', x_test=''):

    if len(x_test)>1:
        xScaler = MinMaxScaler()
        xScaler.fit(x_test.reshape(-1,x_test.shape[1]))
        xTrSc1 = xScaler.transform(x_test.reshape(-1,x_test.shape[1]))
        preds_KNEF = model.predict(xTrSc1[0:len(xTrSc1)])
        preds_KNEF = np.round(preds_KNEF, 4)
        out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

    else:
        xScaler = MinMaxScaler()
        xScaler.fit(x_test)
        xTrSc1 = xScaler.transform(x_test)
        preds_KNEF = model.predict(xTrSc1[0:1])
        out_KNEF = np.round(preds_KNEF, 4)
        out_KNEF = out_KNEF[0]

    return out_KNEF

# функция прогноза KPEF
def preds_KPEF(model='', x_test=''):

    if len(x_test)>1:
        preds_KPEF = model.predict(x_test)
        preds_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = pd.DataFrame(preds_KPEF, columns=['KPEF'])

    else:
        preds_KPEF = model.predict(x_test)
        out_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = out_KPEF[0]

    return out_KPEF

st.write('---')
if result:
    st.subheader('Результат классификации')

    def out_cols():
        if soldatov_cb and not baseline_cb:
            out_collectors = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
            # st.write(out_soldatov_collectors)
        elif baseline_cb and not soldatov_cb:
            out_collectors = preds_argmax_collectors(model=loaded_model_collectors, x_test=predict_collectors)
            # st.write(out_baseline_collectors)
        elif soldatov_cb and baseline_cb:
            collectors = []
            out_soldatov_collectors = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
            out_baseline_collectors = preds_argmax_collectors(model=loaded_model_collectors, x_test=predict_collectors)

            for i in range(len(out_soldatov_collectors)):
                if out_soldatov_collectors[0][i] == out_baseline_collectors[0][i]:
                    collectors.append(out_soldatov_collectors[0][i])
                else:
                    collectors.append(out_soldatov_collectors[0][i])

            out_collectors = pd.DataFrame(collectors, columns=['Коллектор'])

    out_collectors = out_cols()
    
    if knef_radio:
        out_novikov_KNEF = preds_KNEF(model=loaded_model_KNEF, x_test=predict_KNEF)
        # st.write(out_novikov_KNEF)

    if kpef_radio:
        out_fadeev_KPEF = preds_KPEF(model=loaded_model_KPEF, x_test=predict_collectors)
        # st.write(out_fadeev_KPEF)


    if uploaded_file is not None:
        out_all = pd.concat([df, out_collectors, out_novikov_KNEF, out_fadeev_KPEF], axis=1)
    else:
        out_all = pd.DataFrame(predict_KNEF)
        out_all['Коллекторы'] = out_collectors
        out_all['KNEF'] = out_novikov_KNEF
        out_all['KPEF'] = out_fadeev_KPEF
        # out_all = pd.DataFrame([predict_KNEF, out_collectors, out_novikov_KNEF, out_fadeev_KPEF])
    st.write(out_all)
