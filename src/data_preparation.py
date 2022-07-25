import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_birth_data():
    data = pd.read_csv('../../data/birth_data.csv')
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data['ids'] = np.arange(1, data.shape[0]+1)
    data['weekday'] = data.day_of_week.apply(lambda x: 1 if x in [1,2,3,4,5] else 0)
    data['monday'] = data.day_of_week.apply(lambda x: 1 if x != 1 else 0)
    data['seasons'] = data.month.apply(set_season)

    s = StandardScaler()
    y = data.births
    y = np.reshape(y.to_numpy(), (y.shape[0],1))
    data['normalised_births'] = s.fit_transform(y)

    return data


def set_season(x):
    if x in [3,4,5]:
        return 1
    if x in [6,7,8]:
        return 2
    if x in [9,10,11]:
        return 3 
    else:
        return 4


def train_test_save(data):
    df_train, df_test = train_test_split(data, test_size=0.3, random_state=42)
    df_train = df_train.sort_values(by='ids')
    df_test = df_test.sort_values(by='ids')

    s = StandardScaler()
    y_train = df_train.births
    y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0],1))
    y_train = s.fit_transform(y_train)
    df_train['births'] = y_train
    y_test = df_test.births
    y_test = np.reshape(y_test.to_numpy(), (y_test.shape[0],1))
    y_test = s.transform(y_test)
    df_test['births'] = y_test

    df_train.to_csv("../../data/train.csv", index=False)
    df_test.to_csv("../../data/test.csv", index=False)

    # incase we want to use them again later to convert back to original feature space
    scaling_params = pd.DataFrame({'Mean': s.mean_, 'Std': s.scale_})
    scaling_params.to_csv("../data/scaling_params.csv", index=False)


def separate_data(data, weekdays=None, train_test=1):
    if weekdays == 'weekdays':
        data = data.loc[data.weekday==1]
    if weekdays == 'weekends':
        data = data.loc[data.weekday==0]

    x = data.ids
    if train_test == 1:
        y = data.births
    else:
        y = data.normalised_births
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])

    return x, y


def separate_data_with_monday_flag(data):
    x = data.ids
    m = data.monday
    y = data.births
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    m = tf.cast(m, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])
    m = tf.reshape(m, [m.shape[0],1])

    return x, y, m




    
    