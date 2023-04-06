#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


df = pd.read_csv('6-described_data.csv')


# In[4]:


import numpy as np

def clean_dataset(data):
    """
    Remove inputs that contain NaN, infinity or large values for type('float64') in a dataset.

    Args:
    data (numpy array): dataset to be cleaned

    Returns:
    numpy array: cleaned dataset
    """
    mask = np.isnan(data) | np.isinf(data) | (np.abs(data) > np.finfo(np.float64).max)
    data_cleaned = data[~mask.any(axis=1)]
    return data_cleaned


# In[5]:


# clean the dataset
data_cleaned = clean_dataset(df)


# In[6]:


# split the cleaned dataset into input features and output features
X_cleaned = data_cleaned.drop('pIC50', axis=1)
Y_cleaned = data_cleaned.pIC50


# In[8]:


from sklearn.model_selection import train_test_split
# split the cleaned data into training and testing sets
X_train_cleaned, X_test_cleaned, Y_train_cleaned, Y_test_cleaned = train_test_split(X_cleaned, Y_cleaned, test_size=0.2)


# In[9]:


# create a random forest model using the cleaned dataset
model_cleaned = RandomForestRegressor(n_estimators=100)
model_cleaned.fit(X_train_cleaned, Y_train_cleaned)


# In[11]:


from sklearn.model_selection import GridSearchCV

# define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None]
}


# In[13]:


# perform grid search with cross-validation
grid_search = GridSearchCV(model_cleaned, param_grid=param_grid, cv=5)
grid_search.fit(X_train_cleaned, Y_train_cleaned)


# In[14]:


# print the best hyperparameters and the corresponding score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[15]:


import matplotlib.pyplot as plt

# extract the hyperparameter values and the corresponding scores
n_estimators = [params['n_estimators'] for params in grid_search.cv_results_['params']]
max_depths = [params['max_depth'] for params in grid_search.cv_results_['params']]
scores = grid_search.cv_results_['mean_test_score']

# create a 2D plot of the hyperparameter search results
fig, ax = plt.subplots()
for n in param_grid['n_estimators']:
    ax.plot(max_depths, scores[n_estimators.index(n)*len(max_depths):(n_estimators.index(n)+1)*len(max_depths)], label=f"n_estimators={n}")
ax.set_xlabel('max_depth')
ax.set_ylabel('score')
ax.legend()
plt.show()


# In[16]:


# define hyperparameters to test
n_estimators = [50, 100, 150, 200]
max_depth = [10, 20, 30, 40, 50]
param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}


# In[17]:


# perform grid search
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_cleaned, Y_train_cleaned)


# In[18]:


# extract results from grid search
scores = grid_search.cv_results_['mean_test_score']
scores = np.array(scores).reshape(len(n_estimators), len(max_depth))


# In[20]:




