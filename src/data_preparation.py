import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_birth_data():
    data = pd.read_csv('../../data/birth_data.csv')
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data['m-y'] = data['date'].dt.date.apply(lambda x: x.strftime('%Y-%m'))
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


def train_test_save(df):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    df_train = df_train.sort_values(by='ids')
    df_test = df_test.sort_values(by='ids')

    s = StandardScaler()
    y_train = df_train.births
    y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0],1))
    y_train = s.fit_transform(y_train)
    df_train.loc[:, 'births'] = y_train
    y_test = df_test.births
    y_test = np.reshape(y_test.to_numpy(), (y_test.shape[0],1))
    y_test = s.transform(y_test)
    df_test.loc[:, 'births'] = y_test

    df_train.to_csv("../../data/train.csv", index=False)
    df_test.to_csv("../../data/test.csv", index=False)

    # incase we want to use them again later to convert back to original feature space
    scaling_params = pd.DataFrame({'Mean': s.mean_, 'Std': s.scale_})
    scaling_params.to_csv("../data/scaling_params.csv", index=False)

def weekday_train_test_save(df):
    df_weekdays = df[df.weekday == 1]
    weekday_train, weekday_test = train_test_split(df_weekdays, test_size=0.3, random_state=42)
    weekday_train = weekday_train.sort_values(by='ids')
    weekday_test = weekday_test.sort_values(by='ids')
    s = StandardScaler()
    y_train = weekday_train.births
    y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0],1))
    y_train = s.fit_transform(y_train)
    weekday_train.loc[:, 'births'] = y_train
    y_test = weekday_test.births
    y_test = np.reshape(y_test.to_numpy(), (y_test.shape[0],1))
    y_test = s.transform(y_test)
    weekday_test.loc[:, 'births'] = y_test
    weekday_train.to_csv("../../data/weekday_train.csv", index=False)
    weekday_test.to_csv("../../data/weekday_test.csv", index=False)

    df_weekends = df[df.weekday == 0]
    weekend_train, weekend_test = train_test_split(df_weekends, test_size=0.3, random_state=42)
    weekend_train = weekend_train.sort_values(by='ids')
    weekend_test = weekend_test.sort_values(by='ids')
    s = StandardScaler()
    y_train = weekend_train.births
    y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0],1))
    y_train = s.fit_transform(y_train)
    weekend_train.loc[:, 'births'] = y_train
    y_test = weekend_test.births
    y_test = np.reshape(y_test.to_numpy(), (y_test.shape[0],1))
    y_test = s.transform(y_test)
    weekend_test.loc[:, 'births'] = y_test
    weekend_train.to_csv("../../data/weekend_train.csv", index=False)
    weekend_test.to_csv("../../data/weekend_test.csv", index=False)


def train_test_normalise(train_df, test_df):
    s = StandardScaler()
    y_train = train_df.births
    y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0],1))
    y_train = s.fit_transform(y_train)
    train_df = train_df.assign(births=y_train)
    y_test = test_df.births
    y_test = np.reshape(y_test.to_numpy(), (y_test.shape[0],1))
    y_test = s.transform(y_test)
    test_df = test_df.assign(births=y_test)
    return train_df, test_df


def separate_data(df, train_test=1):
    x = df.ids
    if train_test == 1:
        y = df.births
    else:
        y = df.normalised_births
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])

    return x, y


def separate_data_with_monday_flag(df):
    x = df.ids
    m = df.monday
    y = df.normalised_births
    
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    m = tf.cast(m, tf.float64)
    x = tf.reshape(x, [x.shape[0],1])
    y = tf.reshape(y, [y.shape[0],1])
    m = tf.reshape(m, [m.shape[0],1])

    return x, y, m




    
    