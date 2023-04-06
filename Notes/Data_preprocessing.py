"""
Cleaning, transforming, and normalizing the dataset are crucial steps in preparing the data for machine learning models. Here are some common techniques and examples for each step:

1. Cleaning the Dataset
Cleaning the dataset involves removing or correcting errors, missing values, and inconsistent data. Here are some techniques for cleaning the dataset:
Removing duplicates: Duplicate data can bias the analysis, so it is important to remove them. You can use the drop_duplicates() method to remove duplicates based on specific columns or all columns.
"""
# Example

import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')

# Remove duplicates based on all columns
df.drop_duplicates(inplace=True)

"""
Handling missing values: Missing data can cause problems in machine learning models, so it is important to handle them properly. You can fill missing values with the mean, median, or mode of the column or use more sophisticated methods such as regression or imputation.
"""

# Fill missing values with the mean of the column
df['column'].fillna(df['column'].mean(), inplace=True)

"""
Handling outliers: Outliers can skew the data and affect the accuracy of the model. You can remove outliers or transform them using techniques such as logarithmic scaling or box-cox transformation.
"""
import numpy as np

# Remove outliers using the Z-score method
z = np.abs(stats.zscore(df['column']))
df = df[(z < 3)]

# Transform outliers using the logarithmic scaling method
df['column'] = np.log(df['column'])

"""
2. Transforming the Dataset
Transforming the dataset involves converting the data into a more suitable format for machine learning models. Here are some techniques for transforming the dataset:
Encoding categorical variables: Machine learning models require numerical input, so you need to encode categorical variables into numerical variables. You can use techniques such as one-hot encoding, ordinal encoding, or target encoding.
"""
import category_encoders as ce

# One-hot encode a categorical variable
encoder = ce.OneHotEncoder(cols=['column'])
df = encoder.fit_transform(df)

"""
Scaling numerical variables: Machine learning models can be sensitive to the scale of the input variables, so you need to scale numerical variables into a common scale. You can use techniques such as min-max scaling or standardization.
"""
# Example 

from sklearn.preprocessing import MinMaxScaler

# Min-max scale a numerical variable
scaler = MinMaxScaler()
df['column'] = scaler.fit_transform(df[['column']])

"""
Feature engineering: Feature engineering involves creating new features from existing features to improve the performance of the model. You can create new features using mathematical operations, domain knowledge, or external data sources.
"""
# Example

# Create a new feature by combining two existing features
df['new_column'] = df['column_1'] + df['column_2']

"""
3. Normalizing the Dataset
Normalizing the dataset involves rescaling the data to a common range to improve the performance of the model. Here are some techniques for normalizing the dataset:
Min-max scaling: Min-max scaling rescales the data to a common range of values, usually between 0 and 1.
"""
# Min-max scale a numerical variable
df['column'] = (df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())

""" Standardization: Standardization rescales the data to have a mean of 0 and a standard deviation of 1.
"""
# Example

# Standardize a numerical variable
df['column'] = (df['column'] - df['column'].mean


