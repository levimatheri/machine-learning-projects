import numpy as np
import pandas as pd
from sklearn import model_selection
from cross_validation import cross_validate
from predict import *


def init():
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv('iris.data', names=names)

    # plot(dataset)
    # Split validation set
    array = dataset.values

    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

    # crossValidate
    # cross_validate(X_train, Y_train)

    # predict
    # predict(X_train, Y_train, X_validation, Y_validation)
    predict_class([[5.9, 3.0, 5.1, 1.8]])
