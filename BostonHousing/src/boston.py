# Author: Levi Muriuki

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def get_data(filePath):
    return pd.read_csv(filePath)


# remove skewness

def remove_skew(dataframe):
    for i in dataframe.columns:
        if dataframe[i].skew() >= 0.30:
            dataframe[i] = np.log1p(dataframe[i])
        elif dataframe[i].skew() <= -0.30:
            dataframe[i] = np.square(dataframe[i])
    return dataframe


# normalize

def normalize(dataframe):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)


# model

def model(inputs, num_features):
    # hidden layer
    W_1 = tf.Variable(tf.random_normal([num_features, 5], 0, 0.1, dtype=tf.float32))
    b_1 = tf.Variable(tf.zeros([5], dtype=tf.float32))
    layer_1 = tf.add(tf.matmul(inputs, W_1), b_1)
    layer_1_result = tf.nn.relu(layer_1)

    # hidden layer 2
    W_2 = tf.Variable(tf.random_normal([8, 5], 0, 0.1, dtype=tf.float32))
    b_2 = tf.Variable(tf.zeros([5], dtype=tf.float32))
    layer_2 = tf.add(tf.matmul(layer_1_result, W_2), b_2)
    layer_2_result = tf.nn.sigmoid(layer_2)

    # output layer
    W_o = tf.Variable(tf.random_normal([5, 1], 0, 0.1, dtype=tf.float32))
    b_o = tf.Variable(tf.zeros([1], dtype=tf.float32))
    layer_o = tf.add(tf.matmul(layer_2_result, W_o), b_o)
    layer_o_result = tf.nn.sigmoid(layer_o)

    return layer_o_result

