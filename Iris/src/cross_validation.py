from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

scoring = 'accuracy'
models = [('LR', LogisticRegression()), ('KNN', KNeighborsClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]


def cross_validate(X_train, Y_train):
    # try out the models
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)

        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        msg = "%s: %f" % (name, cv_results.mean())

        print(msg)
