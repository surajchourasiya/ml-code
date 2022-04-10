# 1900357 Abhijeet Kumar cse3 6th sem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

da=pd.read_csv('suv_data.csv')
x=da.iloc[:,[2,3]].values
y=da.iloc[:,4].values

import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
d=StandardScaler()
x_train=d.fit_transform(x_train)
x_test=d.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
m=LogisticRegression()
m.fit(x_train,y_train)
y_pred=m.predict()

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc*100)

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
a=x['Age']
plt.scatter(a,y)
plt.scatter(x['EstinatedSalary'],y)

