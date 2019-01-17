from sklearn import metrics
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

scoring = 'accuracy'
models = [('LR', LogisticRegression()),  ('RF', RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1))]#('KNN', KNeighborsClassifier())]


def test_models(X_train, Y_train):
    for name, model in models:
        # # fit model
        model.fit(X_train, Y_train)

        # make predictions
        preds = model.predict(X_train)

        # print accuracy
        accuracy = metrics.accuracy_score(Y_train, preds)
        print("Accuracy: %s" % "{0:.3%}".format(accuracy))

        kfold = model_selection.KFold(n_splits=5)

        results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

        msg = '%s: %f' % (name, results.mean() * 100)

        print(msg)
