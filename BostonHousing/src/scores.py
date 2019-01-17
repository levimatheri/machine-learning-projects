# Author: Levi Muriuki

from sklearn import metrics


def print_metrics(y_test, y_pred):
    r2 = metrics.r2_score(y_test, y_pred)
    print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
    print("r2 score: ", r2)
