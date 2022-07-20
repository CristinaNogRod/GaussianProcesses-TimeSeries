import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def get_birth_data():
    data = pd.read_csv('../data/birth_data.csv')
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data['ids'] = np.arange(1, data.shape[0]+1)
    data['weekday'] = data.day_of_week.apply(lambda x: 1 if x in [1,2,3,4,5] else 0)
    data['monday'] = data.day_of_week.apply(lambda x: 1 if x != 1 else 0)
    # data['births_relative100'] = data.births.apply(lambda x: x/np.mean(data.births)*100)
    data['seasons'] = data.month.apply(set_season)

    y = data.births
    s = StandardScaler()
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
    

def plot_data(x, y):
    plt.figure(figsize=(20,7))
    plt.plot(x, y, '.')
    plt.xlabel('Date')
    plt.ylabel('Number of births')
    plt.title('Births per year')
    plt.show()


def separate_data(data, normalised=None, weekdays=None):
    if weekdays:
        data = data.loc[data.weekday==1]
    x = data.ids
    y = data.births
    if normalised:
        y = data.normalised_births
    else :
        y = y - np.mean(y)
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])

    return x, y

def separate_data_with_monday_flag(data, normalised=None):
    x = data.ids
    m = data.monday
    y = data.births
    if normalised:
        y = data.normalised_births
    else :
        y = y - np.mean(y)
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    m = tf.cast(m, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])
    m = tf.reshape(m, [m.shape[0],1])

    return x, y, m