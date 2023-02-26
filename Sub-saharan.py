
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


path="C:/Users/muya/Desktop/Sub-saharan.csv"

dataset = pd.read_csv(path)

print(dataset.shape)
print(dataset.head())

X=dataset[['avg_temp','rainfall']]
Y=dataset[['yield']]

x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=1)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)

# prediction
y_pred=reg.predict(x_test)
print('Predicted Production:',y_pred,'\n')

# Coefficients
# print('\nCoefficients: ', reg.coef_,'\n')


array=dataset.values
X=array[:,3:5]
Y=array[:,2]

predictions=reg.predict([[16.37,1],[1485,0]])
predictions