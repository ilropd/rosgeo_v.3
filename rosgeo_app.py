import streamlit as st
import tensorflow
import openpyxl
# import xlsxwriter
# import xlrd
# import joblib

import pickle
import requests
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np

st.title('ОПРЕДЕЛЕНИЕ КОНВЕКТОРА')
st.write('---')

st.sidebar.header('Ввод данных для обработки')
uploaded_file = st.sidebar.file_uploader(label='Выберите файл в формате xls для обработки')

cols_collectors = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
cols_KNEF = ['ГЛУБИНА', 'GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
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
        print(predict_KNEF[0:3])

else:
    def accept_user_data():
        st.sidebar.write('')
        st.sidebar.write('или введите данные для вручную')
        depth_korr = st.sidebar.number_input('ГЛУБИНА')
        ggkp_korr = st.sidebar.number_input('GGKP_korr')
        gk_korr = st.sidebar.number_input('GK_korr')
        pe_korr = st.sidebar.number_input('PE_korr')
        ds_korr = st.sidebar.number_input('DS_korr')
        dtp_korr = st.sidebar.number_input('DTP_korr')
        wi_korr = st.sidebar.number_input('Wi_korr')
        bk_korr = st.sidebar.number_input('BK_korr')
        bmk_korr = st.sidebar.number_input('BMK_korr')
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

    input_data_KNEF = accept_user_data()
    input_data_collectors = input_data_KNEF.drop(columns=['ГЛУБИНА'],axis=1)


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
    st.write(input_data_KNEF)


# # загрузка базовой модели для распознавания коллекторов
# json_file = open('venv/Models/js_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights('venv/Models/weights.h5')
# print('Loaded model from disk')


# loaded_model_collectors = pickle.load(open('venv/Models/COLLECTORS/drivemodel_RF_Collectors_model.sav', 'rb'))


url_collectors = 'https://disk.yandex.ru/d/YHsDyy5f5dyxuw'
file_collectors = requests.get(url_collectors)


url_KNEF_model = 'https://disk.yandex.ru/d/Z2dff0Ga0Z-T2w'
file_KNEF_model = requests.get(url_KNEF_model)

url_KNEF_weights = 'https://disk.yandex.ru/d/fnXKBVh1CuN7yQ'
file_KNEF_weights = requests.get(url_KNEF_weights)

url_KPEF_model = 'https://disk.yandex.ru/d/4x-g44cBVuMBcQ'
file_KPEF_model = requests.get(url_KPEF_model)

url_KPEF_weights = 'https://disk.yandex.ru/d/ndQX0CNiwfO4uA'
file_KPEF_weights = requests.get(url_KPEF_weights)


with open(file_collectors, 'rb') as file:
    loaded_model_collectors = pickle.load(file)

json_file_KNEF = open(file_KNEF_model, 'r')
loaded_model_json_KNEF = json_file_KNEF.read()
json_file_KNEF.close()
loaded_model_KNEF = model_from_json(loaded_model_json_KNEF)
loaded_model_KNEF.load_weights(file_KNEF_weights)
print('Loaded model KNEF from disk')

json_file_KPEF = open(file_KPEF_model, 'r')
loaded_model_json_KPEF = json_file_KPEF.read()
json_file_KPEF.close()
loaded_model_KPEF = model_from_json(loaded_model_json_KPEF)
loaded_model_KPEF.load_weights(file_KPEF_weights)
print('Loaded model KNEF from disk')

result = st.button('Классифицировать')
if result:
    st.write('Результат классификации')
    if uploaded_file is not None:

        # # предсказание коллектора с использованием базовой модели
        # preds_collector = loaded_model.predict(predict_collectors)
        # pred_args_collector = np.argmax(preds_collector, axis=1)
        # out_collectors = args_to_types(pred_args_collector)
        # # out_collectors = pd.DataFrame(out_collectors, columns=['Коллектор'])

        preds_collector = loaded_model_collectors.predict(predict_collectors)
        pred_args_collector = np.argmax(preds_collector, axis=1)
        out_collectors = args_to_types(pred_args_collector)


        xScaler = MinMaxScaler()
        xScaler.fit(predict_KNEF.reshape(-1,predict_KNEF.shape[1]))
        xTrSc1 = xScaler.transform(predict_KNEF.reshape(-1,predict_KNEF.shape[1]))
        preds_KNEF = loaded_model_KNEF.predict(xTrSc1[0:len(xTrSc1)])
        preds_KNEF = np.round(preds_KNEF, 4)
        out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

        preds_KPEF = loaded_model_KPEF.predict(predict_collectors)
        preds_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = pd.DataFrame(preds_KPEF, columns=['KPEF'])

        out_all = pd.concat([df, out_collectors, out_KNEF, out_KPEF], axis=1)

        st.write(out_all)

        out_csv = out_all.to_csv()
        download_file = st.download_button('Сохранить', data=out_csv)

    else:
        preds_collector = loaded_model_collectors.predict(input_data_collectors)
        preds_collector_args = np.argmax(preds_collector, axis=1)
        out_collectors = args_to_types(preds_collector_args)

        xScaler = MinMaxScaler()
        xScaler.fit(input_data_KNEF)
        xTrSc1 = xScaler.transform(input_data_KNEF)

        preds_KNEF = loaded_model_KNEF.predict(xTrSc1[0:1])
        preds_KNEF = np.round(preds_KNEF, 4)

        preds_KPEF = loaded_model_KPEF.predict(input_data_collectors)
        preds_KPEF = np.round(preds_KPEF, 4)


        st.write(preds_KNEF)
        st.write(f'Коллектор: {out_collectors[0][0]}. KNEF: {preds_KNEF[0]}. KPEF: {preds_KPEF[0]}')

