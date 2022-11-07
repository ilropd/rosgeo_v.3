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

# подгружаем библиотеку генератора для работы модели коллектров Германа Суслина
from Generator import Generator

# переменные для сохрнения выбора моделей и дальнейшего использования в имени сохраняемого файла
colectors_radio_name = ''
knef_radio_name = ''
kpef_radio_name = ''

# функция генератора для модели Германа Суслина
def get_x(db, length=20):
    form = db.shape[0]-length
    x = [db[i:i + length] for i in range(form)]
    return np.array(x)

# делаем сайдбар с выбором моделей
st.sidebar.header('ВЫБОР МОДЕЛЕЙ ДЛЯ ПРОГНОЗИРОВАНИЯ ГЕОДАННЫХ')

# выбор моделей для предсказания коллекторов делаем чекбоксами, так как будет реализована логика предсказания коллектора
# несколькими моделями, а затем выбор наиблее часто предсказанного как верного
st.sidebar.subheader('Прогнозирование типа коллектора')
# модель 1 - Багурин М.
# модель 2 - Каргальцев В.
# модель 3 - Кононов А.
# модель 4 - Солдатов А.
# интеграционная модель - объединяет в себе модели 1,2 и 4; 3 модель исключена, так как нет argmax
collectors_radio = st.sidebar.radio('Модели Коллекторов', ('модель 1 (Багурин)', 'модель 2 (Каргальцев)', 'модель 3 (Кононов)', 'модель 4 (Солдатов)', 'модель 5 (Суслин)', 'интеграционная модель'))
st.sidebar.write('---')

# выбор моделей KNEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Прогнозирование KNEF')
# модель 1 - Мартынович С.
# модель 2 - Новиков А. (ilro)
# модель 3 - Новиков А.
# модель 4 - Шахлин В.
knef_radio = st.sidebar.radio('Модели KNEF', ('модель 1 (Мартынович)', 'модель 2 (noname)', 'модель 3 (Новиков)', 'модель 4 (Шахлин)'))
st.sidebar.write('---')

# выбор моделей KPEF делаем радиокнопками, так как предсказание будет осуществляться только по одной модели, в отличие
# от предсказания типа коллектора
st.sidebar.subheader('Прогнозирование KPEF')
# модель 1 - Фадеев Ю.
# модель 2 - Шахлин В.
kpef_radio = st.sidebar.radio('Выберите одну из моделей', ('модель 1 (Фадеев)', 'модель 2 (Шахлин)'))

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
                    if (('gk' in j.lower()) and ('ggkp' not in j.lower())) and ('ggkp' in i.lower()):
                        pass
                    else:
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

    # модель Германа Суслина
    json_file_collectors = open('Models/COLLECTORS/Suslin/Conv1d_incr_neurons_consistent_n20_suslin_collector_model.json', 'r')
    loaded_model_json_collectors = json_file_collectors.read()
    json_file_collectors.close()
    loaded_model_suslin_collectors = model_from_json(loaded_model_json_collectors)
    loaded_model_suslin_collectors.load_weights('Models/COLLECTORS/Suslin/Conv1d_incr_neurons_consistent_n20_suslin_collector_model.h5')
    print('Loaded model Suslin COLLECTORS from disk')

    # МОДЕЛИ РАСПОЗНАВАНИЯ KNEF

    # модель Степана Мартыновича
    json_file_KNEF = open('Models/KNEF/Martynovich/Martynovich_final_KNEF_best.json', 'r')
    loaded_model_json_KNEF = json_file_KNEF.read()
    json_file_KNEF.close()
    loaded_model_Martynovich_KNEF = model_from_json(loaded_model_json_KNEF)
    loaded_model_Martynovich_KNEF.load_weights('Models/KNEF/Martynovich/Martynovich_final_KNEF_weights_best.h5')
    print('Loaded model KNEF Martynovich from disk')

    # модель Алексея Новикова
    json_file_Novikov_KNEF = open('Models/KNEF/Novikov/model_Novikov_var3_KNEF_80_without0_model.json', 'r')
    loaded_model_json_Novikov_KNEF = json_file_Novikov_KNEF.read()
    json_file_Novikov_KNEF.close()
    loaded_model_Novikov_KNEF = model_from_json(loaded_model_json_Novikov_KNEF)
    loaded_model_Novikov_KNEF.load_weights('Models/KNEF/Novikov/model_Novikov_var3_KNEF_80_without0_weights.h5')
    print('Loaded model Novikov KNEF from disk')

    # модель Виталия Шахлина
    with urllib.request.urlopen('http://ilro.ru/KNEF/Shakhlin/gradientboosting_shakhlin-KNEF_weights.pkl') as url_shakhlin_knef:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_shakhlin_knef:
            shutil.copyfileobj(url_shakhlin_knef, tmp_shakhlin_knef)

    with open(tmp_shakhlin_knef.name, 'rb') as f_shakhlin_knef:
        loaded_model_shakhlin_knef = pickle.load(f_shakhlin_knef)

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

    return loaded_model_soldatov_collectors, loaded_model_bagurin_collectors, loaded_model_kargaltsev_collectors, loaded_model_suslin_collectors, \
           loaded_model_kononov_collectors, loaded_model_KNEF, loaded_model_Martynovich_KNEF, loaded_model_Novikov_KNEF, loaded_model_shakhlin_knef, loaded_model_KPEF, \
           loaded_model_shakhlin_KPEF

loaded_model_soldatov_collectors, loaded_model_bagurin_collectors, loaded_model_kargaltsev_collectors, loaded_model_suslin_collectors, \
loaded_model_kononov_collectors, loaded_model_KNEF, loaded_model_Martynovich_KNEF, loaded_model_Novikov_KNEF, loaded_model_shakhlin_knef, loaded_model_KPEF, \
loaded_model_shakhlin_KPEF = load_models()

result = st.button('Классифицировать')

# функция предсказания типа коллектора на основе выбранной модели и исходных данных
def preds_argmax_collectors(model='', x_test=''):
    ficha = np.load('Models/COLLECTORS/Kargaltsev/ficha.npy')
    if len(x_test)>1:
        if model is loaded_model_kargaltsev_collectors:
            for i in range(5):
                x_test[:,i] = x_test[:,i] - ficha[i]
            preds_collectors = model.predict(x_test)
            preds_collectors_noargmax = preds_collectors

        elif model is loaded_model_kononov_collectors:
            out_collectors = preds_collectors.astype(int)

        elif model is loaded_model_suslin_collectors:
            preds_20 = loaded_model_bagurin_collectors.predict(x_test)
            preds_collectors_noargmax = preds_20
            preds_20 = np.argmax(preds_20, axis=1)
            preds_20 = args_to_types(preds_20)
            preds_20 = preds_20[0][0]
            preds_20_out = preds_20[0][0][:20]
            preds_20_out = np.array(preds_20_out)

            # убираем столбец, который не участвует в генераторе
            x_data = np.delete(x_test, 3, axis=1)
            x_data2 = get_x(x_data)

            preds_collectors = model.predict(x_data2)
            preds_collectors_noargmax = preds_collectors
            pred_args_collector = np.argmax(preds_collectors, axis=1)
            out_collectors = args_to_types(pred_args_collector)
            out_collectors = np.array(out_collectors[0][0])

            out_collectors = np.concatenate([preds_20_out, out_collectors], axis=1)
            out_collectors = pd.DataFrame(out_collectors)
        else:
            preds_collectors = model.predict(x_test)
            preds_collectors_noargmax = preds_collectors
            pred_args_collector = np.argmax(preds_collectors, axis=1)
            out_collectors = args_to_types(pred_args_collector)

    else:
        if model is loaded_model_suslin_collectors:
            st.write('*Модель 5 не может быть использована для классификации одной строки данных. Классификация будет проведена с использванием Модель 2.*')
            model = loaded_model_kargaltsev_collectors

        if model is loaded_model_kargaltsev_collectors:
            x_test = x_test.to_numpy()
            for i in range(5):
                x_test[:,i] = x_test[:,i] - ficha[i]
            x_test = pd.DataFrame(x_test, columns=cols_collectors)
            preds_collectors = model.predict(x_test)
            preds_collectors_noargmax = preds_collectors
        else:
            preds_collectors = model.predict(x_test)
            preds_collectors_noargmax = preds_collectors

        if model is loaded_model_kononov_collectors:
            out_collectors = preds_collectors.astype(int)
        else:
            pred_args_collector = np.argmax(preds_collectors, axis=1)
            out_collectors = args_to_types(pred_args_collector)
            out_collectors = out_collectors[0][0]

    return out_collectors, preds_collectors_noargmax

# функция прогноза KNEF
def preds_KNEF(model='', x_test='', x_kpef='', x_col=''):

    if len(x_test)>1:

        if model is loaded_model_Novikov_KNEF:
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

        elif model is loaded_model_Martynovich_KNEF:
            x_col = np.array(x_col).reshape(-1,1)
            x_kpef = np.array(x_kpef)
            X_val_knef = np.concatenate([x_test, x_col, x_kpef], axis=1)
            xScaler = MinMaxScaler()
            xScaler.fit(X_val_knef.reshape(-1, X_val_knef.shape[1]))
            xTrSc1 = xScaler.transform(X_val_knef.reshape(-1, X_val_knef.shape[1]))
            preds_KNEF = model.predict(xTrSc1)
            preds_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

        elif model is loaded_model_shakhlin_knef:
            preds_KNEF = model.predict(x_test)
            preds_KNEF = np.exp(preds_KNEF)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

        else:
            xScaler = MinMaxScaler()
            xScaler.fit(x_test.reshape(-1,x_test.shape[1]))
            xTrSc1 = xScaler.transform(x_test.reshape(-1,x_test.shape[1]))
            preds_KNEF = model.predict(xTrSc1)
            preds_KNEF = np.round(preds_KNEF, 4)
            out_KNEF = pd.DataFrame(preds_KNEF, columns=['KNEF'])

    else:
        if model is loaded_model_Novikov_KNEF:
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

        elif model is loaded_model_Martynovich_KNEF:
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

        elif model is loaded_model_shakhlin_knef:
            preds_KNEF = model.predict(x_test)
            preds_KNEF = np.exp(preds_KNEF)
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
        if model is loaded_model_shakhlin_KPEF:
#             st.write('shhhhhh')
            preds_KPEF = model.predict(x_test)
            preds_KPEF = np.exp(preds_KPEF)
        else:
            preds_KPEF = model.predict(x_test)
            preds_KPEF = np.round(preds_KPEF, 4)
#         preds_KPEF = model.predict(x_test)
#         preds_KPEF = np.exp(preds_KPEF)
#         preds_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = pd.DataFrame(preds_KPEF, columns=['KPEF'])

    else:
        if model is loaded_model_shakhlin_KPEF:
            st.write('shhhhhh')
            preds_KPEF = model.predict(x_test)
            preds_KPEF = np.exp(preds_KPEF)
            out_KPEF = np.round(preds_KPEF, 4)
        else:
            preds_KPEF = model.predict(x_test)
            out_KPEF = np.round(preds_KPEF, 4)
#         preds_KPEF = model.predict(x_test)
#         preds_KPEF = np.exp(preds_KPEF)
#         out_KPEF = np.round(preds_KPEF, 4)
        out_KPEF = out_KPEF[0]

    return out_KPEF

st.write('---')
if result:
    st.subheader('Результат классификации')

    def out_cols():
        if collectors_radio == 'модель 4 (Солдтов)':
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)
        elif collectors_radio == 'модель 1 (Багурин)':
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_bagurin_collectors, x_test=predict_collectors)
        elif collectors_radio == 'модель 2 (Каргальцев)':
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_kargaltsev_collectors, x_test=predict_collectors)
        elif collectors_radio == 'модель 3 (Кононов)':
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_kononov_collectors, x_test=predict_collectors)
        elif collectors_radio == 'модель 5 (Суслин)':
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_suslin_collectors, x_test=predict_collectors)
        elif collectors_radio == 'интеграционная модель':
            out_1, out_noargmax_1 = preds_argmax_collectors(model=loaded_model_bagurin_collectors, x_test=predict_collectors)
            out_2, out_noargmax_2 = preds_argmax_collectors(model=loaded_model_kargaltsev_collectors, x_test=predict_collectors)
            out_4, out_noargmax_4 = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)

            out_noargmax_1 = out_noargmax_1*0.93
            out_noargmax_2 = out_noargmax_2*0.939
            out_noargmax_4 = out_noargmax_4*0.9159

            out_collectors_noargmax = out_noargmax_1 + out_noargmax_2 + out_noargmax_4

            out_collector = np.argmax(out_collectors_noargmax, axis=1)

            if uploaded_file is not None:
                out_collector = args_to_types(out_collector)
            else:
                out_collector = args_to_types(out_collector)
                out_collector = out_collector[0][0]

        else:
            out_collector, out_collectors_noargmax = preds_argmax_collectors(model=loaded_model_soldatov_collectors, x_test=predict_collectors)

        return out_collector, out_collectors_noargmax

    out_collectors, out_collectors_noargmax = out_cols()

    if kpef_radio == 'модель 1 (Фадеев)':
        out_KPEF = preds_KPEF(model=loaded_model_KPEF, x_test=predict_collectors)
        # st.write(out_fadeev_KPEF)
    elif kpef_radio == 'модель 2 (Шахлин)':
        out_KPEF = preds_KPEF(model=loaded_model_shakhlin_KPEF, x_test=predict_collectors)

    if knef_radio == 'модель 1 (Мартынович)':
        out_KNEF = preds_KNEF(model=loaded_model_Martynovich_KNEF, x_test=predict_KNEF, x_kpef=out_KPEF, x_col=out_collectors)
    elif knef_radio == 'модель 3 (Новиков)':
        out_KNEF = preds_KNEF(model=loaded_model_Novikov_KNEF, x_test=predict_KNEF, x_kpef=out_KPEF)
    elif knef_radio == 'модель 2 (noname)':
        out_KNEF = preds_KNEF(model=loaded_model_KNEF, x_test=predict_KNEF)
    elif knef_radio == 'модель 4 (Шахлин)':
        out_KNEF = preds_KNEF(model=loaded_model_shakhlin_knef, x_test=predict_collectors)

        # st.write(out_novikov_KNEF)

    if uploaded_file is not None:
        df = df.round({'ГЛУБИНА': 3, 'GGKP': 4, 'GK': 4, 'PE': 4, 'DS': 4, 'DTP': 4, 'Wi': 4, 'BK': 4, 'BMK': 4})
        # out_all = pd.concat([df, out_collectors, out_novikov_KNEF, out_fadeev_KPEF], axis=1)
        out_all = pd.DataFrame(df)
        out_all['Коллекторы'] = out_collectors
        out_all['KNEF'] = out_KNEF.round(4)
        out_all['KPEF'] = out_KPEF.round(4)

    else:
        predict_KNEF = predict_KNEF.round({'ГЛУБИНА': 3, 'GGKP': 4, 'GK': 4, 'PE': 4, 'DS': 4, 'DTP': 4, 'Wi': 4, 'BK': 4, 'BMK': 4})
        out_all = pd.DataFrame(predict_KNEF)
        out_all['Коллекторы'] = out_collectors
        out_all['KNEF'] = out_KNEF.round(4)
        out_all['KPEF'] = out_KPEF.round(4)

    st.write(out_all)

    if uploaded_file is not None:
        if len(uploaded_file.name) < 10:
            uploaded_file_name = uploaded_file.name
        else:
            uploaded_file_name =   uploaded_file.name[0:10]
        collectors_radio_name = 'collectors-' + collectors_radio
        knef_radio_name = 'knef-' + knef_radio
        kpef_radio_name = 'kpef-' + kpef_radio
        file_name_save = '_'.join([uploaded_file_name, collectors_radio_name, knef_radio_name, kpef_radio_name])
    else:
        collectors_radio_name = 'collectors-' + collectors_radio
        knef_radio_name = 'knef-' + knef_radio
        kpef_radio_name = 'kpef-' + kpef_radio
        file_name_save = '_'.join(['Predict', collectors_radio_name, knef_radio_name, kpef_radio_name])

    col_txt, col_csv, col_excel, col_no1 = st.columns(4, gap='small')

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

    with col_txt:
        st.download_button(label='📥 Сохранить в TXT',
                                data=out_csv,
                                file_name=file_name_save+'.txt')
    with col_csv:
        st.download_button(label='📥 Сохранить в CSV',
                                data=out_csv,
                                file_name=file_name_save+'.csv')
    with col_excel:
        st.download_button(label='📥 Сохранить в Excel',
                                data=df_xlsx ,
                                file_name=file_name_save+'.xlsx')
