# 1900357 ABHIJEET KUMAR cse3 6th sem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

data=pd.read_csv("IceCreamData.csv")
data.info()
x=data.iloc[:,:-1]
y=data.iloc[:,-1] 

plt.scatter(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
a=model.predict([[32]])
print(a)
b=model.predict([[10]])
print(b)


 