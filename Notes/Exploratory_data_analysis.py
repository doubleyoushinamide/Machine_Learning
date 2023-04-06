"""
Exploratory data analysis (EDA) is an important step in machine learning that involves exploring the data using statistical techniques and visualization tools to identify patterns and relationships. Here are some techniques and tools that you can use in Python:

1. Descriptive Statistics
Descriptive statistics are used to summarize the distribution of the data and to identify any anomalies or patterns. Here are some common measures of central tendency and dispersion that you can calculate:
Mean: The mean is the average value of the data.
"""
import numpy as np

# Calculate the mean of a column
mean = np.mean(df['column'])

# Median: The median is the middle value of the data.

# Calculate the median of a column
median = np.median(df['column'])

# Mode: The mode is the most common value of the data.

from statistics import mode

# Calculate the mode of a column
mode = mode(df['column'])

# Standard deviation: The standard deviation is a measure of the spread of the data.

# Calculate the standard deviation of a column
std = np.std(df['column'])

""" 2. Visualization Tools
Visualization tools are used to explore the data graphically and to identify any patterns or relationships. Here are some common visualization tools that you can use:
Histogram: A histogram is used to show the distribution of a numerical variable.
"""

import matplotlib.pyplot as plt

# Create a histogram of a column
plt.hist(df['column'], bins=10)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: A scatter plot is used to show the relationship between two numerical variables.

# Create a scatter plot of two columns
plt.scatter(df['column_1'], df['column_2'])
plt.xlabel('Column 1 Name')
plt.ylabel('Column 2 Name')
plt.show()

# Box plot: A box plot is used to show the distribution of a numerical variable and to identify any outliers.

# Create a box plot of a column
plt.boxplot(df['column'])
plt.ylabel('Column Name')
plt.show()

# Heat map: A heat map is used to show the correlation between numerical variables.

import seaborn as sns

# Create a heat map of the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()













