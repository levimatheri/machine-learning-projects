# NON-LINEAR KERNEL

#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

df = pd.read_csv(url, names=cols)

#%%
X = df.drop('Class', axis=1)
y = df['Class']
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#%%
from sklearn.svm import SVC
svc = SVC(kernel='poly', degree=8)
svc.fit(X_train, y_train)
#%%
y_pred = svc.predict(X_test)
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# GAUSSIAN KERNEL
#%%
svc1 = SVC(kernel='rbf')
svc1.fit(X_train, y_train)

y_pred1 = svc1.predict(X_test)

print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
#%%

