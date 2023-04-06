import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

try:
    # Load data from csv file
    data = pd.read_csv('input_file.csv')
except FileNotFoundError:
    print("File not found.")

    # Exit program if file not found
    sys.exit()

# Splitting input variables (SMILE, 2D, 3D descriptors) and target variable (activity)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split the dataset into training set(80%) and test set(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest Regressor object
rf_regressor = RandomForestRegressor()

# Use GridSearchCV to find best Hyperparameters for Random Forest Model
params = {
    "n_estimators": [10, 50, 100],
    "criterion": ["mse", "mae"],
    "max_depth": [None, 5, 10],
    "max_features": [5, 10, X.shape[1]],
}

grid_search_rf = GridSearchCV(
    rf_regressor,
    param_grid=params,
    cv=5,
    scoring="neg_mean_squared_error",
)

grid_search_rf.fit(X_train, y_train)

print("Best parameters:", grid_search_rf.best_params_)

# Train Random Forest Regressor using best hyperparameters
rf_regressor = RandomForestRegressor(**grid_search_rf.best_params_)
rf_regressor.fit(X_train, y_train)

# Predicting results using trained model for both train and test datasets
y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

# Calculate R-squared score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R-squared score for train set:", r2_train)
print("R-squared score for test set:", r2_test)

# Cross Validation Mean Squared Error
cv_scores = cross_val_score(
    rf_regressor,
    X_train,
    y_train,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)

# Overfitting Improvement
overfit_improvement = np.mean(cv_scores) - mean_squared_error(y_test, y_test_pred)

# Plot predicted vs observed activity for test dataset
plt.scatter(y_test, y_test_pred, color='blue')
plt.xlabel('True Activity')
plt.ylabel('Predicted Activity')
plt.title('Random Forest Regression')
plt.savefig('predicted_vs_observed.png')
plt.show()

"""
In this program, we first load the SMILE input descriptors and output activity values from the CSV file using pandas. We then create a random forest model and fit it with the training data. Next, we use this model to predict the activity values for the input data and calculate the r-squared score and mean squared error. Finally, we plot the predicted and actual activity values and save the plot in the working directory using matplotlib.

Note: You may need to install some packages before running this code. You can do so by running !pip install <package name> in the notebook or command line.
"""
