from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def predict(X_train, Y_train, X_validation, Y_validation):
    svm = SVC()
    svm.fit(X_train, Y_train)

    predictions = svm.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    save_model('iris_final_model.sav', svm)


def save_model(filename, model):
    # save model to disk
    print("Saving model to disk...")
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    print("Loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def predict_class(attributes):
    model = load_model('iris_final_model.sav')
    print(model.predict(attributes))
