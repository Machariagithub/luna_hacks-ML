import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:/Users/muya/Desktop/weather.csv')

print(dataset.shape)

print(dataset.describe())

dataset.plot(x='MinTemp',y='MaxTemp', style='o')
plt.title('MinTemp Vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.displot(dataset['MaxTemp'])
plt.show()

# Data Splicing
x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)#training the algorithm

# to retrieve the intercept
print('intercept:',regressor.intercept_)
print('coefficient:',regressor.coef_)

y_pred=regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test.flatten(),'predicted':y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(8,6))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')
plt.show()

plt.scatter(x_test,y_test,color='gray')
plt.plot(x_test,y_pred,color='green',linewidth='1')
plt.show()

print('Mean Absolute Error:',metrics.mean_absolute_error(x_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(x_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(x_test,y_test)))