import streamlit as st
import tensorflow
import openpyxl
# import xlsxwriter
# import xlrd
# import joblib

import pickle
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


json_file_collectors = open('Models/COLLECTORS/Collectors_base_model.json', 'r')
loaded_model_json_collectors = json_file_collectors.read()
json_file_collectors.close()
loaded_model_collectors = model_from_json(loaded_model_json_collectors)
loaded_model_collectors.load_weights('Models/COLLECTORS/Collectors_base_model.h5')
print('Loaded model KNEF from disk')


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
