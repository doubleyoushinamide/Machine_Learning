from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression

# Define function to calculate molecular descriptors
def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol)
    ]
    return descriptors

# Define function to parse data file containing SMILES and target values
def parse_data_file(data_file):
    smiles_list = []
    target_list = []

    with open(data_file, 'r') as f:
        next(f) # skip header line
        for line in f:
            fields = line.rstrip().split(',')
            smiles = fields[0]
            target = float(fields[1])
            smiles_list.append(smiles)
            target_list.append(target)

    return smiles_list, target_list

# Load training data
train_smiles, train_target = parse_data_file('train_data.csv')

# Calculate molecular descriptors for each training molecule
X_train = []
for smiles in train_smiles:
    X_train.append(calc_descriptors(smiles))

# Fit linear regression model to training data
regressor = LinearRegression()
regressor.fit(X_train, train_target)

# Load test data
test_smiles, test_target = parse_data_file('test_data.csv')

# Calculate molecular descriptors for each test molecule
X_test = []
for smiles in test_smiles:
    X_test.append(calc_descriptors(smiles))

# Predict target values for test data using fitted model
test_preds = regressor.predict(X_test)

# Print predicted vs. actual target values for test data
for i in range(len(test_preds)):
    print(f'Predicted: {test_preds[i]:.2f}, Actual: {test_target[i]:.2f}')

"""
This script assumes that the training and test data are stored in CSV files with the first column containing SMILES strings and the second column containing target values. Unlike in the prompt, this script only uses 4 molecular descriptors - the molecular weight, LogP, number of hydrogen bond acceptors, and number of hydrogen bond donors. You can modify the calc_descriptors function to compute additional or different molecular descriptors if desired.

Note that the performance of the linear regression model will depend on the quality and representativeness of the training data, as well as the choice of molecular descriptors. Also, be aware that linear regression assumes a linear relationship between the predictors (molecular descriptors) and the response variable (target value), so it may not be appropriate for all types of data. Other machine learning models, such as decision trees or neural networks, may be better suited for certain QSAR problems.
"""
