"""
Code to calculate RDKit and Mordred descriptors

"""
import pandas as pd
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors
from padelpy import from_mdl
import os
from pathlib import Path
import glob
import csv


def calculate_rdkit_descriptors(mol):
    """
    Calculate RDKit descriptors for a molecule.
    Returns a dictionary with valid numeric descriptors.
    """
    descriptor_functions = {name: function for name, function in Descriptors._descList}
    rdkit_descriptors = {}
    for name, function in descriptor_functions.items():
        try:
            value = function(mol)
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                rdkit_descriptors[f"RDKit_{name}"] = value
        except:
            pass
    return rdkit_descriptors


def calculate_mordred_descriptors(mol):
    """
    Calculate Mordred descriptors for a molecule.
    Returns a dictionary with valid numeric descriptors.
    """
    mordred_calc = Calculator(mordred_descriptors, ignore_3D=False)
    mordred_result = mordred_calc(mol)
    mordred_descriptors_dict = {
        f"Mordred_{str(descriptor)}": float(value)
        for descriptor, value in mordred_result.items()
        if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value)
    }
    return mordred_descriptors_dict


def calculate_padel_descriptors(file_path):
    """
    Calculate PaDEL descriptors from an MDL file.
    Returns a dictionary with valid numeric descriptors.
    """
    try:
        file = os.path.splitext(file_path)[0] + ".mdl"
        if not os.path.exists(file):
            print(f"The file {file} does not exist.")
            return {}

        padel_descriptors = from_mdl(file)
        return {
            f"PaDEL_{k}": float(v)
            for k, v in padel_descriptors.items()
            if v.replace('.', '').isdigit() and not np.isnan(float(v)) and not np.isinf(float(v))
        }
    except Exception as e:
        print(f"Error calculating PaDEL descriptors: {str(e)}")
        return {}


def process_molecule(file_path):
    """
    Process a molecule file and calculate all descriptors.
    Returns a dictionary with all calculated descriptors.
    """
    mol = Chem.MolFromMolFile(file_path)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Calculate Mordred descriptors
        mordred_result = calculate_mordred_descriptors(mol)

        # Calculate RDKit descriptors
        rdkit_result = calculate_rdkit_descriptors(mol)

        # Calculate PaDEL descriptors
        #padel_result = calculate_padel_descriptors(file_path)

        # Combine results
        descriptors_dict = {**mordred_result, **rdkit_result}  # **padel_result
        descriptors_dict['Molecule'] = os.path.splitext(os.path.basename(file_path))[0]

        return descriptors_dict
    else:
        print(f"Could not process molecule {file_path}")
        return None


def start(mol_files):
    """
    Process multiple molecule files and return results.
    Returns a list of descriptor dictionaries.
    """
    results = []
    for file in mol_files:
        result = process_molecule(file)
        if result:
            results.append(result)
    return results


if __name__ == "__main__":
    mol_path = 'mol'
    mol_files = glob.glob(os.path.join(mol_path, '*.mol'))
    results = start(mol_files)

    if results:
        df_descriptors = pd.DataFrame(results)
        if 'Molecule' not in df_descriptors.columns:
            print("Warning: The 'Molecule' column is not in the descriptors. Extracting from file name.")
            df_descriptors['Molecule'] = [Path(f).stem for f in mol_files]
        cols = ['Molecule'] + [col for col in df_descriptors.columns if col != 'Molecule']
        df_descriptors = df_descriptors[cols]
        df_descriptors = df_descriptors.dropna(axis=1, how='any')
        df_descriptors = df_descriptors.loc[:, (df_descriptors.sum() != 0)]
        csv_filename = 'molecular_descriptors_rdkit_mordred_padel.csv'
        df_descriptors.to_csv(csv_filename, index=False)
        print(f"Descriptor results saved in '{csv_filename}'")
        print("\nFirst rows:")
        print(df_descriptors.head())
    else:
        print("Could not process any molecule correctly for descriptors.")
