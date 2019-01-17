from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def predict(X_train, Y_train, X_test, Y_test):
    # logreg = LogisticRegression()
    # logreg.fit(X_train, Y_train)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, Y_train)

    predictions = rfc.predict(X_test)

    print(metrics.accuracy_score(Y_test, predictions))

