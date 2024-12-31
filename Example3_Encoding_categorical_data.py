# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Load the dataset
df = pd.read_csv('titanic.csv')

# Identify the categorical data
categorical_features = ['Sex', 'Embarked', 'Pclass']


# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)

# Apply the fit_transform method on the instance of ColumnTransformer
# Convert the output into a NumPy array
X = ct.fit_transform(df)
X = np.array(X)


# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(df['Survived'])


# Print the updated matrix of features and the dependent variable vector
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)