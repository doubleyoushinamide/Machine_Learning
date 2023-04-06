# Import libraries and load data
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


# Read CSV containing SMILES strings and IC50 labels for each compound
data = pd.read_csv('data.csv')

# Create function to obtain molecular descriptors
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

# Apply the get_descriptors() function to each of the SMILES strings in the dataset
descriptors = np.array([get_descriptors(smi) for smi in data.SMILES])

# Split data into X (descriptors) and Y (IC50 values) arrays.
X = descriptors
Y = data.IC50.values

# Define pipeline for preprocessing and modeling
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression())
])

# Set hyperparameter grids for GridSearchCV
param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__solver': ['saga', 'liblinear']
}

# Perform GridSearchCV with 5-Fold Cross Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)

# Initialize coefficient list to track model convergence
coefs = []

best_model = None # Initialize to None

# Train the model over multiple epochs and track convergence
for epoch in range(10):
    print("Epoch ", epoch)
    grid.fit(X, Y)
    new_model = grid.best_estimator_

    # Check if new model is better than previous best model
    if best_model is None or new_model.score(X, Y) > best_model.score(X, Y):
        best_model = new_model

    # Track the coefficients of the trained model
    coefs.append(best_model.named_steps['model'].coef_.ravel())

    # Stop early if convergence has been reached
    if len(coefs) > 1 and np.allclose(coefs[-1], coefs[-2]):
        print("Convergence reached after", epoch, "epochs")
        break

# Predict IC50 values for the test set using the converged model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
predicted = best_model.predict(X_test)

# Calculate Metrics for test set
r2 = r2_score(Y_test, predicted)
rmse_test = np.sqrt(mean_squared_error(Y_test, predicted))

# Plotting Observed vs Predicted Values
with plt.figure() as fig:
    plt.scatter(Y_test, predicted, alpha=0.5)
    plt.title('Observed vs Predicted')
    plt.xlabel('Observed IC50 values')
    plt.ylabel('Predicted IC50 values')
    fig.savefig(f'observed_vs_predicted_{best_model["model"].__class__.__name__}.png')

# Export CSV containing observed and predicted labels
pd.DataFrame({'Observed IC50 values': Y_test, 'Predicted IC50 values': predicted}).to_csv('observed_vs_predicted.csv', index=False)

# Saving other relevant graphs
# *Descriptors Distribution*
with plt.figure(figsize=(10,6)) as fig:
    ax = plt.hist(np.sum(X, axis=1), bins=100)
    plt.title('Morgan Fingerprint Density')
    plt.xlabel('Descriptor Frequency')
    plt.ylabel('Density')
    fig.savefig('descriptors_density.png')

# *Descriptors Pair Plot*
descriptors_df = pd.DataFrame(data=descriptors)
descriptors_df['IC50'] = Y
sns.pairplot(descriptors_df.sample(500), hue="IC50")
plt.savefig('pairplot.png')

"""
In this updated code, we have added a loop that runs the GridSearchCV function multiple times and tracks the coefficients of the model at each epoch. The loop continues until the difference in model coefficients between consecutive epochs is small enough to indicate convergence. Once convergence is reached, we use the final converged model to predict the IC50 values for the test set.

Note: The number of epochs that need to be run to reach convergence may vary depending on the dataset and the chosen learning rate.
"""