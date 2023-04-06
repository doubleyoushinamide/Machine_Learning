import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# load data from csv file
data = pd.read_csv("data.csv")

# split into input and target variables
inputs = data.iloc[:, 1:-1].values 
targets = data.iloc[:, -1].values

# train test split with 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# scaling input data to zero mean and unit variance
scaler = StandardScaler()  
scaler.fit(X_train)  

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# neural network model initialization 
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000) 

# fit the model to the training data
mlp.fit(X_train, y_train)

# prediction on test data 
y_pred = mlp.predict(X_test) 

# calculate model accuracy (R-squared value)
accuracy = r2_score(y_test, y_pred)

# Scores validation
scores = cross_val_score(mlp, inputs, targets, cv=5) # 5-fold cross validation
print("Cross Validation Score:", np.mean(scores))

#Overfitting improvement
y_train_pred = mlp.predict(X_train)
train_accuracy = r2_score(y_train, y_train_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", accuracy)
print("Improvement from Training to Testing:", accuracy - train_accuracy)

# print out model accuracy
print("Model Accuracy (R-squared):", accuracy)

# plot predicted vs actual values
plt.scatter(y_test, y_pred) 
plt.xlabel("Actual Values") 
plt.ylabel("Predictions") 
plt.title("Neural Network Model")
plt.savefig("predicted_vs_actual.png")
plt.show()

"""
In addition to the corrections, some improvements were made:

Added separate printing statements for training and testing accuracy.
Added the 'testing' label next to Testing Accuracy: for clarity.
Added the R-squared label for the Model Accuracy statement.
Reordered code for better readability.
"""