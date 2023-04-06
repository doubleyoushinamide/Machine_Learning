import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, PandasTools

# load csv file containing SMILES notations of compounds
data = pd.read_csv('compounds.csv')

# get the SMILES from csv file into rdkit mol object, then calculate descriptors
mols = [Chem.MolFromSmiles(smiles) for smiles in data['SMILES']]

# Calculate 2D descriptors and assign to the columns named "Descriptors_2D"
desc_2d = []
for mol in mols:
    desc_2d.append(list(Descriptors._CalcDescriptors(mol)))
data['Descriptors_2D'] = desc_2d

# Calculate 3D descriptors and assign to the columns named "Descriptors_3D"
desc_3d = []
for mol in mols:
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        desc_3d.append(list(mp.CalcDescriptors(mol)))
    except Exception as error:
        # if calculation fails for any molecule, print molecule smiles and error message
        print(f"Error with SMILE: {error}, Skipping. ", mol)
data['Descriptors_3D'] = desc_3d

# print out the descriptors of all the molecules
print(data[['SMILES', 'Descriptors_2D', 'Descriptors_3D']])

"""
Note: The code above will print out the SMILES along with the calculated 2D and 3D descriptors for each compound. You can modify the output format or write it to a new file based on your needs using Pandas.
"""
