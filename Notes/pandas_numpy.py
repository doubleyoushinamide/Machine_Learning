"""
NumPy provides support for large, multi-dimensional arrays and matrices, along with a wide range of mathematical functions to operate on these arrays. Pandas, on the other hand, is a library for data manipulation and analysis. It provides data structures for efficiently storing and manipulating large datasets and supports a wide range of data formats, including CSV, Excel, and SQL databases.

Here are some examples of how you can use Pandas and NumPy in machine learning:
"""
# Numpy examples:
## Creating arrays

import numpy as np

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])

# Create a 2D array
b = np.array([[1, 2], [3, 4], [5, 6]])

# Create an array of zeros
c = np.zeros((3, 3))

# Create an array of ones
d = np.ones((2, 2))

# Create a random array
e = np.random.random((2, 2))

# Performing arithmentic operations:
import numpy as np

# Add two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b

# Multiply two arrays
d = np.array([[1, 2], [3, 4]])
e = np.array([[5, 6], [7, 8]])
f = np.dot(d, e)

# slicing and indexing array import numpy as np

# Slicing a 1D array
a = np.array([1, 2, 3, 4, 5])
b = a[1:4] # returns [2, 3, 4]

# Slicing a 2D array
c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = c[1:, :2] # returns [[4, 5], [7, 8]]

# Indexing with boolean arrays
e = np.array([1, 2, 3, 4, 5])
f = e[e > 3] # returns [4, 5]

"""
Pandas is a widely-used Python library for data manipulation and analysis. It is particularly useful in machine learning applications, as it allows you to load, clean, and transform data before training a model.

Here are some of the key use cases and functions of pandas in machine learning, along with examples:

1. Data Loading and Cleaning
Pandas provides a wide range of functions for loading data from different sources such as CSV, Excel, SQL databases, and web APIs. You can use pandas to load data into a dataframe, which is a 2-dimensional table-like data structure with rows and columns. Once you have loaded the data, you can use pandas functions to clean the data by removing null or duplicate values, renaming columns, and handling missing data.
"""

# Example
import pandas as pd

# Load data from CSV file
df = pd.read_csv('data.csv')

# Remove rows with missing values
df = df.dropna()

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})

# Remove duplicate rows
df = df.drop_duplicates()

"""
1. Data Transformation
Pandas provides powerful functions for data transformation such as filtering, sorting, grouping, and aggregating data. You can use pandas to transform data to match the requirements of your machine learning model, such as scaling numeric features or encoding categorical variables.
"""
# Example

# Select rows with a certain condition
df_filtered = df[df['column_name'] > 10]

# Sort data by column
df_sorted = df.sort_values(by='column_name')

# Group data by a column and calculate the mean
df_grouped = df.groupby('group_column').mean()

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['categorical_column'])

"""
3. Data Visualization
Pandas provides easy-to-use functions for data visualization such as bar plots, line plots, scatter plots, and histograms. You can use pandas to visualize your data and gain insights into its distribution, relationships, and trends.
"""
# Example

import matplotlib.pyplot as plt

# Plot a histogram of a numeric column
df['numeric_column'].hist()
plt.show()

# Plot a scatter plot of two columns
df.plot.scatter(x='column_1', y='column_2')
plt.show()

# Plot a bar chart of a categorical column
df['categorical_column'].value_counts().plot.bar()
plt.show()

"""
4. Feature Engineering
Feature engineering involves creating new features from existing data to improve the performance of a machine learning model. Pandas provides functions to create new features from existing ones, such as calculating ratios, aggregating data, or applying mathematical functions.
"""

# Calculate a new feature as a ratio of two columns
df['ratio'] = df['column_1'] / df['column_2']

# Aggregate data and create a new feature
df_grouped = df.groupby('group_column')['numeric_column'].agg(['sum', 'mean', 'max'])
df_grouped = df_grouped.rename(columns={'sum': 'sum_numeric_column', 'mean': 'mean_numeric_column', 'max': 'max_numeric_column'})

# Apply a mathematical function to a column and create a new feature
df['log_column'] = np.log(df['numeric_column'])

"""
5. Data Preprocessing
Data preprocessing involves transforming raw data into a format suitable for machine learning. Pandas provides functions to normalize, scale, and encode data, as well as to split data into training and test sets.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train

# MATPLOTLIB

"""
Matplotlib is a Python library for data visualization. It provides a wide range of functions for creating different types of plots, such as line plots, scatter plots, bar plots, and histograms. Matplotlib is often used in machine learning applications to visualize data and model performance.

Here are some of the key use cases and functions of Matplotlib in machine learning, along with examples:

1. Data Visualization
Matplotlib provides a wide range of functions for data visualization, such as line plots, scatter plots, bar plots, and histograms. You can use Matplotlib to visualize your data and gain insights into its distribution, relationships, and trends.
"""
# Examples

import matplotlib.pyplot as plt

# Plot a line plot of a numeric column
plt.plot(df['numeric_column'])
plt.show()

# Plot a scatter plot of two columns
plt.scatter(df['column_1'], df['column_2'])
plt.show()

# Plot a bar chart of a categorical column
plt.bar(df['categorical_column'].value_counts().index, df['categorical_column'].value_counts())
plt.show()

# Plot a histogram of a numeric column
plt.hist(df['numeric_column'])
plt.show()

"""
2. Model Performance Visualization
Matplotlib is often used to visualize the performance of machine learning models, such as accuracy, precision, recall, and F1 score. You can use Matplotlib to plot these metrics over time or compare them across different models.
"""
# Example

# Plot the accuracy of a model over time
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot the precision, recall, and F1 score of a model
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
plt.bar(['precision', 'recall', 'F1 score'], [precision, recall, f1_score])
plt.title('Model Performance')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.show()

"""
3. Data Exploration
Matplotlib can also be used for exploratory data analysis, which involves discovering patterns and relationships in the data. You can use Matplotlib to create scatter matrices, heatmaps, and other types of plots to explore the data.
"""
# Example

import seaborn as sns

# Create a scatter matrix of numeric columns
sns.pairplot(df.select_dtypes(include='number'))
plt.show()

# Create a heatmap of correlations between columns
sns.heatmap(df.corr())
plt.show()

"""
4. Customization
Matplotlib allows you to customize your plots in many ways, such as changing the colors, adding labels and legends, and adjusting the axes. You can use Matplotlib to create plots that are tailored to your specific needs.
"""
# Example

# Change the color of a plot
plt.plot(df['numeric_column'], color='red')
plt.show()

# Add labels and a legend to a plot
plt.plot(df['column_1'], label='column 1')
plt.plot(df['column_2'], label='column 2')
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.legend()
plt.show()

# Adjust the axes limits of a plot
plt.plot(df['numeric_column'])
plt.xlim([0, 100])
plt.ylim([0, 10])
plt.show()

"""
5. Saving Plots
Matplotlib allows you to save your plots in various formats such as PNG, PDF, and SVG. You can use Matplotlib to save your plots for use in reports or presentations.
"""
# Save a plot as a PNG





