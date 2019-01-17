# LINEAR KERNEL

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#%%
bd = pd.read_csv("data_banknote_authentication.csv")

#%%
bd.shape 

#%%
bd.head()

#%%
# preprocessing
X = bd.drop('class', axis=1)
y = bd['class']

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#%%
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
#%%
y_pred = svc.predict(X_test)
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
#%%