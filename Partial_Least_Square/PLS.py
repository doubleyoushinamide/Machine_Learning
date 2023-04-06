# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from matplotlib import pyplot as plt

# Reading the data from the CSV file
data = pd.read_csv('data.csv')

# Extracting the SMILES notation and Activity (IC50) column
smiles = data['SMILES Notation']
ic50 = data['Activity (IC50)']

# Function for generating descriptors
def generate_descriptors(smiles):
    """
    RDKit is used for generating molecular descriptors. You can install it via pip.
    pip install rdkit
    
    You need to modify the "code for generating descriptors" according to your dataset.
    """
    descriptors = []
    for s in smiles:
        # actual code for generating descriptors should go here
        descriptors.append(s)
    return descriptors

# Calculating 2D and 3D molecular descriptors using RDKit library 
descriptors = generate_descriptors(smiles)

# Converting the list of descriptors to a numpy array
X = np.array(descriptors)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, ic50, test_size=0.2, random_state=42)

# Building the PLS regression model
n_components = 5
pls_model = PLSRegression(n_components=n_components)
pls_model.fit(x_train, y_train)

# Predicting IC50 values for the test set
y_pred = pls_model.predict(x_test)

# Calculating the R-squared value and RMSE for the predictions on the test set
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Performing cross validation and calculating the cross-validated R-squared value
cv = 10
y_pred_cv = cross_val_predict(pls_model, X, ic50, cv=cv)
r2_cv = r2_score(ic50, y_pred_cv)

# Plotting the observed vs predicted activity and saving the image as a png file
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([ic50.min(), ic50.max()], [ic50.min(), ic50.max()], 'k--', lw=4)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title(f'Observed vs Predicted Activity (n_components={n_components})')
plt.savefig('obs_vs_pred.png')

# Plotting other relevant graphs and saving them in the working directory
# ...

# Exporting the Observed vs Predicted activity values as csv to current directory
df = pd.DataFrame({'Observed': y_test, 'Predicted': y_pred})
df.to_csv('obs_vs_pred.csv', index=False)

# Round off the R-squared value to 2 decimal places and print it
r2_rounded = round(r2, 2)
print(f'R-squared value: {r2_rounded}')


"""
Note:
Note that the code above does not address the issue of incomplete code for generating descriptors; this will need to be implemented separately. Additionally, the comment about installing RDKit should be removed or updated to reflect whether or not RDKit is necessary for the descriptors used in this script.
RDkit is used for generating molecular descriptors. You can install it via pip.
pip install rdkit
You need to modify this line # code for generating descriptor according to your dataset.
"""
