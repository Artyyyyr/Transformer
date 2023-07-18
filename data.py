import numpy as np
import pandas as pd
from math import e
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

names_columns = ['ID', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']

for i in range(1, 22):
    names_columns.append('sensor_m_' + str(i))


additional = 50


def process_data_step_prediction(raw_data, steps=5):
    dataframe = pd.DataFrame(data=raw_data, columns=names_columns)

    _dataframe = dataframe.groupby(['ID'])['cycle'].max().reset_index(name='END')
    dataframe = pd.merge(_dataframe, dataframe, on='ID')
    dataframe.loc[dataframe['END'] != dataframe['cycle'], 'END'] = 0
    dataframe.loc[dataframe['END'] == dataframe['cycle'], 'END'] = 1

    y = dataframe[['END', 'ID']].values.tolist()
    x = dataframe[dataframe.columns.difference(['END', 'ID', 'cycle'])].values.tolist()

    x = _normalize_data(x, np.load("means_of_data.npy"), np.load("vars_of_data.npy"))

    _y = [[]]
    _x = [[]]

    for i in range(len(y)):
        _x[-1].append(x[i])
        _y[-1].append(y[i][0])
        if y[i][0] == 1:
            if i != len(y) - 1:
                _y.append([])
                _x.append([])

    y = []

    for engine_x in _x:
        y.append(_sigmoid_enhanced(len(engine_x), additional=additional))

    x = _x

    for i in range(len(x)):
        x[i] = tf.Variable([x[i]])
        y[i] = tf.Variable([y[i]])

    return x, y


def _sigmoid_y(a, additional=0):
    sigmoid = []
    for x in range(a + additional):
        bad = 1 / (1 + pow(e, -x))
        sigmoid.append(bad)

    return sigmoid


def _sigmoid_enhanced(a, additional=0):
    sigmoid = []
    for x in range(a + additional):
        bad = 1 / (1 + pow(e,  -(x - a + 14) / 10))
        sigmoid.append(bad)

    return sigmoid


def _get_mean_and_variance(x):
    scaler = StandardScaler()
    scaler.fit_transform(x)

    return scaler.mean_, scaler.var_


def _normalize_data(x, means, variances):
    x = np.array(x).transpose()
    for num, column in enumerate(x):
        column = column - means[num]
        column = column / np.sqrt(variances[num])
        x[num] = column
    x = np.array(x).transpose()
    return x


def get_points(data):
    break_point = len(data[0][0])
    before_bp = np.linspace(0, break_point - 10, 7, dtype=int).tolist()
    bp = np.linspace(break_point - 9, break_point + 9, 3, dtype=int).tolist()
    after_bp = np.linspace(break_point + 10, break_point + 48, 7, dtype=int).tolist()
    res = []
    res.extend(before_bp)
    res.extend(bp)
    res.extend(after_bp)
    return res
