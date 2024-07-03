import os
import subprocess
import argparse
import warnings

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mols2grid
import useful_rdkit_utils as uru
from rdkit import Chem
from rdkit.Chem import PandasTools
import MDAnalysis as mda
import prolif as plf
from rdkit.Chem.rdFMCS import FindMCS

def setup_environment():

    warnings.filterwarnings('ignore')

    # Configuring path variables for OpenEye software
    os.environ["PATH"] += ":/usr/local/openeye/bin"
    os.environ["OE_LICENSE"] = "/home/fts_g_ucla_edu/Projects/oe_license.txt"

    # Setting random seed
    random.seed(42)

def write_molecules(num_mols):

    # Opens fragments.smi to read, creates mol2mol.smi to write
    fragments = open("data/fragments.smi", "r")
    mol2mol = open("data/mol2mol.smi", "w")

    for i in range(num_mols):
        mol = fragments.readline()
        mol2mol.write(mol)

    fragments.close()
    mol2mol.close()

def remove_odd_rings(df):
    ring_system_lookup = uru.RingSystemLookup.default()
    df['ring_systems'] = df.SMILES.apply(ring_system_lookup.process_smiles)
    df[['min_ring','min_freq']] = df.ring_systems.apply(uru.get_min_ring_frequency).to_list()
    df = df.query('min_freq > 100').copy()
    return df.iloc[:,0:3]

def run_reinvent(toml_file):
    # Assuming the reinvent command is installed and accessible
    command = f"reinvent  tomls/{toml_file} --seed 42"
    subprocess.run(command, shell=True)

def run_docking_pipeline(df, generate_ifp):
    # Placeholder for docking pipeline execution
    print("\nRunning docking pipeline...\n")

    # Additional code for docking pipeline can be added here
    subprocess.run(["spruce", "-in", "data/docking/5s18.pdb", "-out", "data/docking/5s18.oedu"], check=True)

    df['Name'] = [f"MOL{i:04d}" for i in range(len(df))]
    df[["SMILES","Name"]].to_csv("data/docking/analogs_rings_ok.smi", sep=" ", header=None, index=False)

    subprocess.run(["/usr/local/openeye/bin/omega2", "-in", "data/docking/analogs_rings_ok.smi", "-out", "data/docking/analogs_rings_ok.oeb", "-strictstereo", "false", "-log", "/dev/null"], check=True)
    subprocess.run(["/usr/local/openeye/bin/hybrid", "-receptor", "data/docking/5s18.oedu", "-dbase", "data/docking/analogs_rings_ok.oeb", "-out", "data/docking/analogs_rings_ok_docked.sdf"], check=True)

    docked_df = PandasTools.LoadSDF("data/docking/analogs_rings_ok_docked.sdf")
    ref_mol = Chem.MolFromMolFile("data/docking/5s18_ligand.sdf")

    rmsd_list = [mcs_rmsd(ref_mol, x) for x in docked_df.ROMol.values]
    docked_df['rmsd'] = rmsd_list

    df_rmsd_ok = docked_df.query("rmsd <= 2").copy()

    PandasTools.WriteSDF(df_rmsd_ok,"data/docking/analogs_rmsd_ok.sdf")

    # Generate columns in dataframe

    mol_ids = df_rmsd_ok['ID'].to_numpy()

    df['Success'] = False

    for index, row in df.iterrows():
        if row['Name'] in mol_ids:
            df['Success'][index] = True

    df = df.merge(docked_df[['ID', 'HYBRID Chemgauss4 score']], left_on='Name', right_on='ID', how='left')
    df.rename(columns={'HYBRID Chemgauss4 score' : 'Docking score'}, inplace=True)

    # If generate_ifp is flagged, write the interaction fingerprint to a csv
    if generate_ifp:
       ifp = write_ifp(df)
       return (df, ifp)
    else:
        return (df, None)


"""
Calculates the RMSD for maximum common substructure between two ligands
"""
def mcs_rmsd(mol_1, mol_2):
    mcs_res = FindMCS([mol_1, mol_2])
    pat = Chem.MolFromSmarts(mcs_res.smartsString)
    match_1 = mol_1.GetSubstructMatches(pat)
    match_2 = mol_2.GetSubstructMatches(pat)
    min_rmsd = 1e6
    for m1 in match_1:
        for m2 in match_2:
            crd_1 = mol_1.GetConformer(0).GetPositions()[list(m1)]
            crd_2 = mol_2.GetConformer(0).GetPositions()[list(m2)]
            rmsd = np.sqrt((((crd_1 - crd_2)**2).sum()).mean())
            min_rmsd = min(min_rmsd, rmsd)
    return min_rmsd


"""
TO DO: Write function that writes interaction fingerprint to a csv
"""
def write_ifp(df):
    pass

def main():
    parser = argparse.ArgumentParser(description="Run reinvent and optionally the docking pipeline")
    parser.add_argument("--toml_file", type=str, help="The path to the TOML file for the reinvent command")
    parser.add_argument("--dock", action="store_true", help="Flag to run the docking pipeline after reinvent")
    parser.add_argument("--generate_ifp", action="store_true", help="Flag to generate the interaction fingerprint as a csv")
    parser.add_argument("--num_mols", type=int, choices=range(1, 236), default=1)
    parser.add_argument("--remove_odd_rings", type=bool, choices=[True, False], default=True, help="Turn off to keep generated molecules with odd ring systems")
    
    args = parser.parse_args()

    # Setup the environment
    setup_environment()

    # Write .smi file with entry molecules
    write_molecules(args.num_mols)

    # Run reinvent with the provided TOML file
    run_reinvent(args.toml_file)

    # Read output from reinvent to dataframe
    df = pd.read_csv('data/sampling.csv')

    # Remove odd ring systems
    df = remove_odd_rings(df)

    # Conditionally run the docking pipeline
    if args.dock:
        df, ifp = run_docking_pipeline(df, args.generate_ifp)
        
    df.to_csv('data/dataframe.csv')


if __name__ == "__main__":
    main()
