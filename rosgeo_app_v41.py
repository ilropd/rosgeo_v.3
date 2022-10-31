import streamlit as st
# import streamlit.components.v1 as components
# import tensorflow
# import openpyxl
# import requests
import urllib.request
import tempfile
import shutil
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
# import io
from io import BytesIO
import pandas as pd
import numpy as np
# import sklearn
# from sklearn_gbmi import *
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.metrics import log_loss

# делаем сайдбар с выбором моделей
st.sidebar.header('ВЫБОР МОДЕЛЕЙ ДЛЯ ПРОГНОЗИРОВАНИЯ ГЕОДАННЫХ')

# выбор моделей для предсказания коллекторов делаем чекбоксами, так как будет реализована логика предсказания коллектора
# несколькими моделями, а затем выбор наиблее часто предсказанного как верного
st.sidebar.subheader('Прогнозирование типа коллектора')
# bagurin_cb = st.sidebar.checkbox('Багурин М.')
# grigorevckiy_cb = st.sidebar.checkbox('Григоревский К.')
collectors_radio = st.sidebar.radio('Модели Коллекторов', ('Багурин М.', 'Каргальцев В.', 'Кононов А.', 'Солдатов А.'))
# bagurin_cb = st.sidebar.checkbox('Багурин М.')
# kargaltsev_cb = st.sidebar.checkbox('Каргальцев В.')
# kononov_cb = st.sidebar.checkbox('Кононов А.')
# soldatov_cb = st.sidebar.checkbox('Солдатов А.', value=True)
st.sidebar.write('---')

# выбор моделей KNEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Прогнозирование KNEF')
# knef_radio = st.sidebar.checkbox('Новиков А. (ilro)', value=True)
knef_radio = st.sidebar.radio('Модели KNEF', ('Мартынович С.', 'Новиков А. (ilro)', 'Новиков А.'))
st.sidebar.write('---')

# выбор моделей KPEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Прогнозирование KPEF')
# kpef_radio = st.sidebar.checkbox('Фадеев Ю.', 'Шахлин В.', value=True)
kpef_radio = st.sidebar.radio('Выберите одну из моделей', ('Фадеев Ю.', 'Шахлин В.'))

# основной блок с выводом информации
st.title('ПРОГНОЗИРОВАНИЕ ГЕОДАННЫХ')
st.write('---')

st.header('Ввод данных для обработки')

# подгрузка файла с примером данных .csv
with open("Downloads/Example_rosgeology.csv", "rb") as file:
    st.download_button(
        label="📥 Скачать пример файла для загрузки в CSV",
        data=file,
        file_name="Example_rosgeology.csv"
        )

# вызываем блок для загрузки файла
uploaded_file = st.file_uploader(label='Выберите файл в формате XLS или CSV для обработки', )

# используем разное количество заголовков колонок, так как какие-то модели работают с 8 столбцами, какие-то с 9-10
cols_collectors = ['GGKP', 'GK', 'PE', 'DS', 'DTP', 'Wi', 'BK', 'BMK']
cols_KNEF = ['ГЛУБИНА', 'GGKP', 'GK', 'PE', 'DS', 'DTP', 'Wi', 'BK', 'BMK']

# осуществляем предобработку данных, загруженных через файл
if uploaded_file is not None:
        st.write('*Файл загружен успешно*')

        if 'csv' in uploaded_file.name:
            df = pd.read_csv(uploaded_file, sep=',', decimal=',', header=0)
            df = df.dropna(axis='index', how='any')

        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

            for i in range(len(df)):
                for j in range(len(df.iloc[i])):
                  if df.iloc[i][j] in cols_KNEF:
                    df.rename(columns=df.iloc[i], inplace = True)

            for k in range(len(df)):
                for l in range(len(df.iloc[k])):
                    if type(df.iloc[k][l]) is str:
                        # print(db.iloc[k][l])
                        df.iloc[k][l]= np.nan

            df = df.dropna(axis='index', how='any')
            df.reset_index(drop=True, inplace=True)

        for i in df.columns.values:
            for j in cols_collectors:
                if (j.lower() in i.lower()) and ('KPEF'.lower() not in i.lower()):
                    df.rename(columns={i: j}, inplace=True)

        for col in df.columns:
            if col not in cols_KNEF:
                    df.pop(col)

        df = df.reindex(columns=cols_KNEF)
        # st.write(df)

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
            depth = st.number_input('ГЛУБИНА')
            ggkp = st.number_input('GGKP')
            gk = st.number_input('GK')

        with col2:
            pe = st.number_input('PE')
            ds = st.number_input('DS')
            dtp = st.number_input('DTP')

        with col3:
            wi = st.number_input('Wi')
            bk = st.number_input('BK')
            bmk = st.number_input('BMK')

        data = {'ГЛУБИНА': depth,
                'GGKP': ggkp,
                'GK': gk,
                'PE': pe,
                'DS': ds,
                'DTP': dtp,
                'Wi': wi,
                'BK': bk,
                'BMK': bmk
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

    with open(tmp_soldatov.name, 'rb') as f_soldatov:
        loaded_model_soldatov_collectors = pickle.load(f_soldatov)

    # модель Антона Кононова
    with urllib.request.urlopen('http://ilro.ru/Collectors/Kononov/BaggingClassifier_RG_kononov_collectors_model.pkl') as url_kononov:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_kononov:
            shutil.copyfileobj(url_kononov, tmp_kononov)

    with open(tmp_kononov.name, 'rb') as f_kononov:
        loaded_model_kononov_collectors = pickle.load(f_kononov)

    # модель Максима Багурина
    json_file_collectors = open('Models/COLLECTORS/Bagurin/bagurin_collectors_93_model.json', 'r')
    loaded_model_json_collectors = json_file_collectors.read()
    json_file_collectors.close()
    loaded_model_bagurin_collectors = model_from_json(loaded_model_json_collectors)
    loaded_model_bagurin_collectors.load_weights('Models/COLLECTORS/Bagurin/bagurin_collectors_93_weights.h5')
    print('Loaded model Bagurin COLLECTORS from disk')

    # модель Владислава Каргальцева
    json_file_collectors = open('Models/COLLECTORS/Kargaltsev/rgmodel19_Kargaltsev_collectors.json', 'r')
    loaded_model_json_collectors = json_file_collectors.read()
    json_file_collectors.close()
    loaded_model_kargaltsev_collectors = model_from_json(loaded_model_json_collectors)
    loaded_model_kargaltsev_collectors.load_weights('Models/COLLECTORS/Kargaltsev/rgmodel19_Kargaltsev_collectors_weights.h5')
    print('Loaded model Kargaltsev COLLECTORS from disk')

    # МОДЕЛИ РАСПОЗНАВАНИЯ KNEF

    # модель Степана Мартыновича
    json_file_KNEF = open('Models/KNEF/Martynovich/KNEF_final_Martynovich_model.json', 'r')
    loaded_model_json_KNEF = json_file_KNEF.read()
    json_file_KNEF.close()
    loaded_model_Martynovich_KNEF = model_from_json(loaded_model_json_KNEF)
    loaded_model_Martynovich_KNEF.load_weights('Models/KNEF/Martynovich/KNEF_final_Martynovich_weights.h5')
    print('Loaded model KNEF Martynovich from disk')

    # модель Алексея Новикова
    json_file_Novikov_KNEF = open('Models/KNEF/Novikov/model_Novikov_var3_KNEF_80_without0_model.json', 'r')
    loaded_model_json_Novikov_KNEF = json_file_Novikov_KNEF.read()
    json_file_Novikov_KNEF.close()
    loaded_model_Novikov_KNEF = model_from_json(loaded_model_json_Novikov_KNEF)
    loaded_model_Novikov_KNEF.load_weights('Models/KNEF/Novikov/model_Novikov_var3_KNEF_80_without0_weights.h5')
    print('Loaded model Novikov KNEF from disk')


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


    # json_file_collectors = open('Models/COLLECTORS/Collectors_base_model.json', 'r')
    # loaded_model_json_collectors = json_file_collectors.read()
    # json_file_collectors.close()
    # loaded_model_collectors = model_from_json(loaded_model_json_collectors)
    # loaded_model_collectors.load_weights('Models/COLLECTORS/Collectors_base_model.h5')
    # print('Loaded model COLLECTORS from disk')


    json_file_KNEF = open('Models/KNEF/model_ilro_KNEF_model.json', 'r')
    loaded_model_json_KNEF = json_file_KNEF.read()
    json_file_KNEF.close()
    loaded_model_KNEF = model_from_json(loaded_model_json_KNEF)
    loaded_model_KNEF.load_weights('Models/KNEF/model_ilro_KNEF_weights.h5')
    print('Loaded model KNEF from disk')

    # МОДЕЛИ РАСПОЗНАВАНИЯ KPEF
    # модель Виталия Шахлина
    # with urllib.request.urlopen('http://ilro.ru/KPEF/Shakhlin/decisiontree_shakhlin-KPEF_weights.pkl') as url_shakhlin:
    with urllib.request.urlopen('http://ilro.ru/KPEF/Shakhlin/gradientboosting_shakhlin-KPEF_weights.pkl') as url_shakhlin:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_shakhlin:
            shutil.copyfileobj(url_shakhlin, tmp_shakhlin)

    with open(tmp_shakhlin.name, 'rb') as f_shakhlin:
        loaded_model_shakhlin_KPEF = pickle.load(f_shakhlin)

    # модель Юрия Фадеева
    json_file_KPEF = open('Models/KPEF/KPEF_baseline_model.json', 'r')
    loaded_model_json_KPEF = json_file_KPEF.read()
    json_file_KPEF.close()
    loaded_model_KPEF = model_from_json(loaded_model_json_KPEF)
    loaded_model_KPEF.load_weights('Models/KPEF/KPEF_baseline_weights.h5')
    print('Loaded model KPEF from disk')

    return loaded_model_soldatov_collectors, loaded_model_bagurin_collectors, loaded_model_kargaltsev_collectors, \
           loaded_model_kononov_collectors, loaded_model_KNEF, loaded_model_Martynovich_KNEF, loaded_model_Novikov_KNEF, loaded_model_KPEF, \
           loaded_model_shakhlin_KPEF

loaded_model_soldatov_collectors, loaded_model_bagurin_collectors, loaded_model_kargaltsev_collectors, \
loaded_model_kononov_collectors, loaded_model_KNEF, loaded_model_Martynovich_KNEF, loaded_model_Novikov_KNEF, loaded_model_KPEF, \
loaded_model_shakhlin_KPEF = load_models()

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
        out_collectors = out_collectors[0][0]

    return out_collectors

# функция прогноза KNEF
def preds_KNEF(model='', x_test='', x_kpef='', x_col=''):

    if len(x_test)>1:

        if knef_radio == 'Новиков А.':
            X_val_knef = np.array(x_kpef).reshape(-1,1)
            xScaler = MinMaxScaler()
            xScaler.fit(x_test.reshape(-1,x_test.shape[1]))
            xValSc = xScaler.transform(x_test.reshape(-1,x_test.shape[1]))
            xValSc1 = xValSc[:, 0:5]
            xValSc2 = xValSc[:, 5:8]
            preds_KNEF = model.predict([xValSc1, xValSc2, X_val_knef])
            preds_KNEF = np.round(((preds_KNEF-0.5)/0.5), 4)
            preds_KNEF = np.round(preds_KNEF, 4)
            # min_max = MinMaxScaler(feature_range=(preds_KNEF.min(), preds_KNEF.max()))
            # preds_KNEF = min_max.fit_transform(preds_KNEF)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])
                # .apply(lambda x: x*0.003/preds_KNEF.min())

        elif knef_radio == 'Мартынович С.':
            x_col = np.array(x_col)
            x_kpef = np.array(x_kpef)
            X_val_knef = np.concatenate([x_test, x_col, x_kpef], axis=1)
            xScaler = MinMaxScaler()
            xScaler.fit(X_val_knef.reshape(-1,X_val_knef.shape[1]))
            xTrSc1 = xScaler.transform(X_val_knef.reshape(-1,X_val_knef.shape[1]))
            preds_KNEF = model.predict(xTrSc1)
            preds_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

        else:
            xScaler = MinMaxScaler()
            xScaler.fit(x_test.reshape(-1,x_test.shape[1]))
            xTrSc1 = xScaler.transform(x_test.reshape(-1,x_test.shape[1]))
            preds_KNEF = model.predict(xTrSc1)
            preds_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

    else:
        if knef_radio == 'Новиков А.':
            x_test = np.array(x_test)
            X_val_kpef = np.array(x_kpef).reshape(-1,1)
            xScaler = MinMaxScaler()
            xScaler.fit(x_test.reshape(-1,x_test.shape[1]))
            xValSc = xScaler.transform(x_test.reshape(-1,x_test.shape[1]))
            st.dataframe(xValSc)
            xValSc1 = xValSc[:, 0:5]
            xValSc2 = xValSc[:, 5:8]
            preds_KNEF = model.predict([xValSc1[0:1], xValSc2[0:1], X_val_kpef[0:1]])
            out_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = (out_KNEF[0]-0.5)/0.5
            # out_KNEF = out_KNEF*1/min(out_KNEF)

        elif knef_radio == 'Мартынович С.':
            x_test = np.array(x_test)
            x_col = np.array([x_col]).reshape(-1,1)
            x_kpef = np.array(x_kpef).reshape(-1,1)
            X_val_knef = np.concatenate([x_test, x_col, x_kpef], axis=1)
            xScaler = MinMaxScaler()
            xScaler.fit(X_val_knef)
            xTrSc1 = xScaler.transform(X_val_knef)
            preds_KNEF = model.predict(xTrSc1[0:1])
            out_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = out_KNEF[0]

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
        # if model == 'loaded_model_shakhlin_KPEF':
        #     st.write('shhhhhh')
        #     preds_KPEF = model.predict(x_test)
        #     preds_KPEF = np.exp(preds_KPEF)
        # else:
        #     preds_KPEF = model.predict(x_test)
        #     preds_KPEF = np.round(preds_KPEF, 4)
        preds_KPEF = model.predict(x_test)
        preds_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = pd.DataFrame(preds_KPEF, columns=['KPEF'])

    else:
        # if model == 'loaded_model_shakhlin_KPEF':
        #     st.write('shhhhhh')
        #     preds_KPEF = model.predict(x_test)
        #     preds_KPEF = np.exp(preds_KPEF)
        #     out_KPEF = np.round(preds_KPEF, 4)
        # else:
        #     preds_KPEF = model.predict(x_test)
        #     out_KPEF = np.round(preds_KPEF, 4)
        preds_KPEF = model.predict(x_test)
        out_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = out_KPEF[0]

    return out_KPEF

st.write('---')
if result:
    st.subheader('Результат классификации')

    def out_cols():
        if knef_radio == 'Солдатов А.':
            out_collector = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
        elif knef_radio == 'Багурин М.':
            out_collector = preds_argmax_collectors(model=loaded_model_bagurin_collectors, x_test=predict_collectors)
        elif knef_radio == 'Каргальцев В.':
            out_collector = preds_argmax_collectors(model=loaded_model_kargaltsev_collectors, x_test=predict_collectors)
        elif knef_radio == 'Кононов А.':
            out_collectors = preds_argmax_collectors(model=loaded_model_kononov_collectors, x_test=predict_collectors)
        else:
            out_collector = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)

        return out_collector
        # if soldatov_cb and not (bagurin_cb or kargaltsev_cb or kononov_cb):
        #     out_collectors = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
        #     # st.write(out_soldatov_collectors)
        # elif bagurin_cb and not (soldatov_cb or kargaltsev_cb or kononov_cb):
        #     out_collectors = preds_argmax_collectors(model=loaded_model_bagurin_collectors, x_test=predict_collectors)
        #     # st.write(out_baseline_collectors)
        # elif kargaltsev_cb and not (soldatov_cb or bagurin_cb or kononov_cb):
        #     out_collectors = preds_argmax_collectors(model=loaded_model_kargaltsev_collectors, x_test=predict_collectors)
        # elif kononov_cb and not (soldatov_cb or bagurin_cb or kargaltsev_cb):
        #     out_collectors = preds_argmax_collectors(model=loaded_model_kononov_collectors, x_test=predict_collectors)
        # else:
        #     out_collectors = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)

        # elif soldatov_cb and (bagurin_cb or kargaltsev_cb):
        #     collectors = []
        #     out_soldatov_collectors = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
        #     out_bagurin_collectors = preds_argmax_collectors(model=loaded_model_bagurin_collectors, x_test=predict_collectors)
        #     out_kargaltsev_collectors = preds_argmax_collectors(model=loaded_model_kargaltsev_collectors, x_test=predict_collectors)
        #
        #     for i in range(len(out_soldatov_collectors)):
        #         if out_soldatov_collectors[0][i] == (out_bagurin_collectors[0][i] or out_kargaltsev_collectors[0][i]):
        #             collectors.append(out_soldatov_collectors[0][i])
        #         else:
        #             collectors.append(out_soldatov_collectors[0][i])

            # out_collectors = pd.DataFrame(collectors, columns=['Коллектор'])

    out_collectors = out_cols()


    if kpef_radio == 'Фадеев Ю.':
        out_KPEF = preds_KPEF(model=loaded_model_KPEF, x_test=predict_collectors)
        # st.write(out_fadeev_KPEF)
    elif kpef_radio == 'Шахлин В.':
        out_KPEF = preds_KPEF(model=loaded_model_shakhlin_KPEF, x_test=predict_collectors)

    if knef_radio == 'Мартынович С.':
        out_KNEF = preds_KNEF(model=loaded_model_Martynovich_KNEF, x_test=predict_KNEF, x_kpef=out_KPEF, x_col=out_collectors)
    elif knef_radio == 'Новиков А.':
        out_KNEF = preds_KNEF(model=loaded_model_Novikov_KNEF, x_test=predict_KNEF, x_kpef=out_KPEF)
    elif knef_radio == 'Новиков А. (ilro)':
        out_KNEF = preds_KNEF(model=loaded_model_KNEF, x_test=predict_KNEF)
        # st.write(out_novikov_KNEF)

    if uploaded_file is not None:
        # out_all = pd.concat([df, out_collectors, out_novikov_KNEF, out_fadeev_KPEF], axis=1)
        out_all = pd.DataFrame(df)
        out_all['Коллекторы'] = out_collectors
        out_all['KNEF'] = out_KNEF
        out_all['KPEF'] = out_KPEF

    else:
        out_all = pd.DataFrame(predict_KNEF)
        out_all['Коллекторы'] = out_collectors
        out_all['KNEF'] = out_KNEF
        out_all['KPEF'] = out_KPEF
        # out_all = pd.DataFrame([predict_KNEF, out_collectors, out_novikov_KNEF, out_fadeev_KPEF])
    st.write(out_all)

    col_csv, col_excel, col_no1, col_no2 = st.columns(4, gap='small')

    out_csv = out_all.to_csv()
    out_xls = pd.DataFrame(out_all)

    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        # worksheet = writer.sheets['Sheet1']
        # format1 = workbook.add_format({'num_format': '0.00'})
        # worksheet.set_column('A:A', None, format1)
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = to_excel(out_xls)

    with col_csv:
        st.download_button(label='📥 Сохранить в CSV',
                                data=out_csv,
                                file_name= 'Rosgeology_prediction.txt')
    with col_excel:
        st.download_button(label='📥 Сохранить в Excel',
                                data=df_xlsx ,
                                file_name= 'Rosgeology_prediction.xlsx')
