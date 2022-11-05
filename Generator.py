from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики

import random

import tensorflow
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # для генерации выборки временных рядов
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, \
    Input, Dense, Dropout, BatchNormalization, \
    Flatten, Conv1D, Conv2D, \
    LSTM  # Стандартные слои

import tensorflow.keras.backend as K

import pandas as pd
import numpy as np

# Список ошибок
range_ggkp_loss = 0.1
range_gk_loss = 1.5
range_pe_loss = 1
range_ds_loss = 50
range_dtp_loss = 20

errors = [range_ggkp_loss, range_gk_loss, range_pe_loss,
          range_ds_loss, range_dtp_loss]


# error_column_inx = [i for i in range(len(x_train[0]) - 3)]
def tpe(y_true, y_pred):
    true_value = y_true
    pred_value = y_pred
    tpe = K.mean(K.abs(true_value - pred_value)/true_value*100)
    return tpe
def adding_error(x: np.array, columns: list, errors: list) -> np.array:
    '''
    Добавляет ошибку к исходным данным и возвращает их
    :param x: входной массив
    :param columns: индексы колонок к которым применяются ошибки
    :param errors: ошибки колонок
    :return: массив данных с ошибками из списка
    '''
    for row in range(len(x)):
        for column in columns:
            x[row][column] = round(x[row][column] + \
                                   random.uniform(-errors[column],
                                                  errors[column]), 3)
    return x


class Generator(tensorflow.keras.utils.Sequence):
    '''
    Генератор батчей с x - x_len*batch_size и y - batch_size
    '''

    def __init__(self, x_data,
                 y_data_coll,
                 y_data_rest,
                 length,
                 batch_size,
                 x_columns,
                 y_columns,
                 only_colls=True):
        # Инициализируем и записывем переменные батча и сколько
        # понадобится предыдущих значений для предсказания, а также колонки, которые выбраны для анализа
        '''
        :param x_data: np.array
        Входные данные
        :param y_data_coll: np.array
        Данные по типу коллектора
        :param y_data_rest: np.array
        Данные по двум оставшимся показателям
        :param length: int
        Длина выборки для предсказания
        :param batch_size: int
        размер батча
        :param x_columns: list of ints [0-8]
        какие колонки задействуются
        :param y_columns: list of inst [0-3]
        пока не используется - но выбор колонок
        '''
        self.x_data = x_data[:, x_columns]
        self.y_data_coll = y_data_coll
        self.y_data_rest = y_data_rest
        self.x_columns = x_columns
        # self.y_columns = y_columns
        self.batch_size = batch_size
        self.length = length
        self.norm_fit = []
        self.norm_y_fit = []
        self.only_colls = only_colls
        self.len_norm = len(x_columns)
    def info(self):
        '''
        Вывод информации о датафрейма
        '''
        # print(self.df.head())
        print(f'\nРазмер: {self.x_data.shape, self.y_data_coll.shape, self.y_data_rest.shape}')

    def add_error(self, errors_indx, errors_value):
        # Добавляем ошибки к исходным данным
        self.len_norm = len(errors_indx)
        for row in range(len(self.x_data)):
            for column in range(len(errors_indx)):
                self.x_data[row, column] = round(self.x_data[row, column] + \
                                                 random.uniform(-errors_value[column],
                                                                errors_value[column]), 3)
        print(self.x_data.shape)

    def normalize(self, columns, norm_y = False):
        # нормализация каждого столбца данных
        for i in columns:
            x = self.x_data[:, i].reshape(-1, 1)
            xScaler = StandardScaler()
            xScaler.fit(x)
            self.norm_fit.append(xScaler)
            self.x_data[:, i] = np.array(xScaler.transform(x)).reshape(-1)
        if norm_y:
            for i in range(len(self.y_data_rest[0])):
                y = self.y_data_rest[:, i].reshape(-1, 1)
                yScaler = StandardScaler()
                yScaler.fit(y)
                self.y_data_rest[:, i] = np.array(yScaler.transform(y)).reshape(-1)
                self.norm_y_fit.append(yScaler)
        # возвращает список нормализаторов, чтобы потом нормализировать
        # тестовые данные
        return self.norm_fit, self.norm_y_fit

    def normalize_test(self, norm_fit, norm_y_fit = None, norm_y = False):
        for i in range(len(norm_fit)):
            x = self.x_data[:, i].reshape(-1, 1)
            self.x_data[:, i] = np.array(norm_fit[i].transform(x)).reshape(-1)
        if norm_y:
            for i in range(len(norm_y_fit)):
                y = self.y_data_rest[:, i].reshape(-1, 1)
                self.y_data_rest[:, i] = np.array(norm_y_fit[i].transform(y)).reshape(-1)

    def __get_data(self, x_batch, y_batch_coll, y_batch_rest):
        # Разбиваем наш батч на сеты
        # Определим максимальный индекс
        form = x_batch.shape[0] - self.length
        x = [x_batch[i:i + self.length] for i in range(form)]
        y_coll = [y_batch_coll[i] for i in range(form)]
        y_rest = [y_batch_rest[i] for i in range(form)]
        return np.array(x), np.array(y_coll), np.array(y_rest)

    def __len__(self):
        return (self.x_data.shape[0] - self.length) // self.batch_size

    def __getitem__(self, index):
        # Формирование выборки батчей
        # Берём значения от 0 до размера батча + длина выборки предсказания
        x_batch = self.x_data[index * self.batch_size:
                              (index + 1) * self.batch_size + self.length - 1]

        y_batch_coll = self.y_data_coll[index * self.batch_size + self.length - 1:
                                        (index + 1) * self.batch_size + self.length - 1]

        y_batch_rest = self.y_data_rest[index * self.batch_size + self.length - 1:
                                        (index + 1) * self.batch_size + self.length - 1]
        # print(y_batch_rest.shape, y_batch_coll.shape, x_batch.shape)
        x, y_coll, y_rest = self.__get_data(x_batch, y_batch_coll, y_batch_rest)

        if self.only_colls == True:
            return x, y_coll
        else:
            return x, y_rest
        # return x, [y_coll, y_rest]

class Generator2d(Generator):
    def __get_data(self, x_batch, y_batch_coll, y_batch_rest):
        # Разбиваем наш батч на сеты
        # Определим максимальный индекс
        form = x_batch.shape[0] - self.length
        x = [x_batch[i:i + self.length] for i in range(form)]
        y_coll = [y_batch_coll[i] for i in range(form)]
        y_rest = [y_batch_rest[i] for i in range(form)]
        return np.expand_dims(x, 2), np.array(y_coll), np.array(y_rest)

class Worker:
    def __init__(self, fname):
        '''
        Инициализация
        '''
        self.df = pd.read_csv(fname, decimal=',')
        self.enc = OneHotEncoder()
    def info(self):
        '''
        Вывод информации о датафрейма
        '''
        print(self.df.head())
        print(f'\nРазмер: {self.df.shape}')

    def get_y_collektors(self):
        '''
        Получение y_data для столбца "Коллекторы"
        '''
        # Преобразование в OHE
        enc = OneHotEncoder()
        y_data_coll = enc.fit_transform(
            self.df['Коллекторы'].values.reshape(-1, 1)
        ).toarray().astype(np.int16)
        self.enc = enc
        y_data_rest = self.df[['KPEF', 'KNEF']].values.astype(np.float32)
        print(f'Размер: {y_data_coll.shape, y_data_rest.shape}', self.df.columns)
        return y_data_coll, y_data_rest
    def get_enc(self):
        return self.enc
    def get_x_data(self, columns):
        '''
        Получение x_data
        - columns - список столбцов вида ['GGKP_korr', 'GK_korr', 'DTP_korr']
        '''
        get_x_data = self.df[columns].values.astype(np.float32)
        print(f'Размер: {get_x_data.shape}')
        return get_x_data

# Функция рассчёта точности/ошибки
def accuracy_calculate(model, x_val, y_val, colls = True, scaler = None):
    right_answer = []
    predVal = model.predict(x_val)
    if colls:
        # Рассчёт точности для модели предсказания коллекторов
        for i, x in enumerate(predVal):
            if np.argmax(x) == np.argmax(y_val[i]):
                right_answer.append([np.argmax(x), i])
        right_answer = np.array(right_answer)
        accuracy = len(right_answer) / len(y_val)
        # print('accuracy', accuracy)
        return accuracy
    else:
        # Рассчёт средней ошибки для модели предсказания KNEF/KPEF
        for i, x in enumerate(predVal):
            loss = abs((x[0]-y_val[i, 0]))
            right_answer.append(loss)
        loss = sum(right_answer)/len(right_answer)
        # loss = scaler.transform(loss)
        return loss

