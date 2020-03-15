

@author: Tanya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('advertising.csv')
X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predict the values for test set
y_pred=regressor.predict(X_test)

#visualize training set

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('TV Ads vs Sales (Training Set)')
plt.xlabel('TV Ads')
plt.ylabel('Sales')
plt.show()

#visualize test set

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('TV Ads vs Sales (Test Set)')
plt.xlabel('TV Ads')
plt.ylabel('Sales')
plt.show()
