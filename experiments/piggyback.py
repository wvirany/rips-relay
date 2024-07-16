import subprocess
import os

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(42)

import pandas as pd

from crem.crem import grow_mol, mutate_mol
crem_db = '../crem_db/crem_db2.5.db'

import mols2grid

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import PandasTools
from rdkit import DataStructs

def tanimoto_similarity(smi_1, smi_2, use_counts=True):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,countSimulation=True)
    mol_1 = Chem.MolFromSmiles(smi_1)
    mol_2 = Chem.MolFromSmiles(smi_2)
    if use_counts:
        fp_1 = rdFingerprintGenerator.GetCountFPs([mol_1])[0]
        fp_2 = rdFingerprintGenerator.GetCountFPs([mol_2])[0]
    else:
        fp_1 = rdFingerprintGenerator.GetFPs([mol_1])[0]
        fp_2 = rdFingerprintGenerator.GetFPs([mol_2])[0]
    return DataStructs.TanimotoSimilarity(fp_1, fp_2)

def piggyback(initial, lead, mol_list, model='reinvent'):

    initial_mol = Chem.MolFromSmiles(initial)
    results = {}    # Temporary dict to store results per-run

    if model == 'reinvent': # Generate analogs w/ REINVENT

        # Change directory to run python script
        %cd ..

        arg1 = f'--input_frag'
        subprocess.run(['python3', 'generate_analogs.py', arg1, initial],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
        
        # Change directory back to that of the current notebook
        %cd experiments
        
        # Read dataframe
        df = pd.read_csv('data/dataframe.csv')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    elif model == 'crem':   # Generate analogs w/ CReM
        
        out_list = []
        mutate_list = list(mutate_mol(initial_mol, db_name=crem_db, return_mol=False))

        for idx, analog in enumerate(mutate_list):
            out_list.append([analog, initial])

        df = pd.DataFrame(out_list, columns=["SMILES","Input_SMILES"])

    else:   # Raise an error if an invalid model is entered
        raise Exception('Invalid Model')
    
    # Remove duplicate values
    df.drop_duplicates(inplace=True, ignore_index=True, subset=['SMILES'])
    
    # Remove the initial fragment from the generated distribution
    if initial in df['SMILES'].values:

        index = df.loc[df['Tanimoto'] == 1].index[0]
        df.drop(index)

    df.round(3)

    # Compute similarities to lead molecule
    similarities_to_lead = [tanimoto_similarity(analog, lead, True) for analog in df['SMILES'].values]
    df['sim_to_lead'] = similarities_to_lead
    df.sort_values('sim_to_lead', ascending=False, inplace=True)

    # Calculate mean and max Tanimoto similarities
    results['mean'], results['max'], results['num_analogs'] = [df['sim_to_lead'].mean()], [df['sim_to_lead'].max()], len(df)
    

    if len(df) > 0:
        # Find SMILES string corresponding to best value
        best = df['SMILES'].values[0]
        
        # Add best to list of piggybacked molecules
        mol_list.append(Chem.MolFromSmiles(best))
    else:
        best = None

    return best, mol_list, results