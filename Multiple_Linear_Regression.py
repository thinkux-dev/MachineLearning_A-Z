# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Simeple_Linear_Regression import X_train, X_test, y_train, regressor, y_pred

# Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# X = iris_df.drop('target', axis=1)
# y = iris_df['target']

print(X)

## Encoding Categorical Data
# Implement an instance of the ColumnTransformer class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)

# Apply the fit_transform method on the instance of ColumnTransformer
# Convert the output into a NumPy array
X = np.array(ct.fit_transform(X))
# print(X)

# # Use LabelEncoder to encode binary categorical data
# le = LabelEncoder()
# y = le.fit_transform(y)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#this round the values to 2 decimal place
np.set_printoptions(precision=2)

# To check the relationship of Vector of predicted profit(y_pred) is closed to vector of real profit(y_test)
# i.e checking if our model is able to return some predictions close to the real profit
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


## Bonus

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000,
# Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
# Ans-> [ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
# 42467.52924853204

# Explanation
# Therefore, the equation of our multiple linear regression model is:
# Profit=86.6×Dummy State1−873×DummyState2+786×DummyState3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

## Important Note: To get these coefficients we called the "coef_" and "intercept_"
# attributes from our regressor object. Attributes in Python are different than methods
# and usually return a simple value or an array of values.