# Importing necessary libraries
import os
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import logging


# Setting up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Reading the input data in csv format
if os.path.isfile("input_data.csv"):
    data = pd.read_csv("input_data.csv")
else:
    logging.error("'input_data.csv' does not exist in the current directory.")
    exit(1)

# Converting SMILES notations to molecular objects
mol_list = []
for mol_smile in data["SMILES"]:
    mol = Chem.MolFromSmiles(mol_smile)
    if not mol:
        logging.warning(f"Unable to convert {mol_smile} to Mol object.")
    else:
        mol_list.append(mol)

# Generating 2D and 3D molecular descriptors
des_list = []
for mol in mol_list:
    # 2D descriptors
    desc_2D = [
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.TPSA(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumValenceElectrons(mol),
    ]

    # 3D descriptors
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        desc_3D = [Descriptors.MolVolume(mol), Descriptors.CalcTPSA(mol)]
    except:
        logging.warning(f"Descriptor calculation failed for something in {mol}. Check for any special characters.")

    # Combining 2D and 3D descriptors
    des = desc_2D + desc_3D
    des_list.append(des)

# Converting descriptor lists to dataframe
descriptors = pd.DataFrame(des_list)

# Merging the descriptor dataframe with activity data
activity = data["Activity"]
data_new = pd.concat([activity, descriptors], axis=1)

# Splitting the data into training and testing sets
X = data_new.iloc[:, 1:]
y = data_new.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the IC50 values using the test set
y_pred = lr.predict(X_test)

# Calculating the regression coefficient (R2) and root mean square error (RMSE)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculating the cross-validation (R2predict)
r2p = np.mean(cross_val_score(lr, X, y, cv=10))

# Creating the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Plotting the Observed vs predicted activity graph from the model and saving in png file
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Observed Activity (IC50)")
plt.ylabel("Predicted Activity (IC50)")
plt.title("Observed vs Predicted Activity (IC50)")
plt.savefig("output/obs_vs_pred.png")
plt.close()


# Exporting the Observed vs Predicted activity values as csv to current directory
df_result = pd.DataFrame({'Observed_Activity': y_test, 'Predicted_Activity': y_pred})
df_result.to_csv('obs_vs_pred.csv', index=False)

# Printing the R2, R2predict and RMSE values
print("Linear Regression R2: %.2f" % r2)
print("Cross-Validation R2: %.2f" % r2p)
print("Linear Regression RMSE: %.2f" % rmse)

"""
This code will generate output for regression coefficient (R2), cross-validation (R2predict) and root mean square error (RMSE).

Note: As the size of the dataset is unknown here, there would be approximations involved. You can adjust them according to your dataset size.
"""
