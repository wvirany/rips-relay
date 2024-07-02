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


def run_reinvent(toml_file):
    # Assuming the reinvent command is installed and accessible
    command = f"reinvent  tomls/{toml_file} --seed 42"
    subprocess.run(command, shell=True)

def run_docking_pipeline(df):
    # Placeholder for docking pipeline execution
    print("Running docking pipeline...")

    # Additional code for docking pipeline can be added here
    subprocess.run(["spruce", "-in", "data/openeye/5s18.pdb", "-out", "data/openeye/5s18.oedu"], check=True)

    df['Name'] = [f"MOL{i:04d}" for i in range(len(df_ok))]
    df[["SMILES","Name"]].to_csv("data/analogs_rings_ok.smi", sep=" ", header=None, index=False)

    subprocess.run(["/usr/local/openeye/bin/omega2", "-in", "data/analogs_rings_ok.smi", "-out", "data/openeye/analogs_rings_ok.oeb", "-strictstereo", "false"], check=True)
    subprocess.run(["/usr/local/openeye/bin/hybrid", "-receptor", "data/5s18.oedu", "-dbase", "data/openeye/analogs_rings_ok.oeb", "-out", "data/analogs_rings_ok_docked.sdf"], check=True)

    docked_df = PandasTools.LoadSDF("data/openeye/analogs_rings_ok_docked.sdf")
    ref_mol = Chem.MolFromMolFile("data/openeye/5s18_ligand.sdf")

    rmsd_list = [mcs_rmsd(ref_mol, x) for x in docked_df.ROMol.values]
    docked_df['rmsd'] = rmsd_list

    df_rmsd_ok = docked_df.query("rmsd <= 2").copy()

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

def main():
    parser = argparse.ArgumentParser(description="Run reinvention and optionally the docking pipeline.")
    parser.add_argument("--toml_file", type=str, help="The path to the TOML file for the reinvent command.")
    parser.add_argument("--dock", action="store_true", help="Flag to run the docking pipeline after reinvention.")
    parser.add_argument("--num_mols", type=int, choices=range(1, 236), default=1)
    
    args = parser.parse_args()

    # Setup the environment
    setup_environment()

    # Write .smi file with entry molecules
    write_molecules(args.num_mols)

    # Run reinvent with the provided TOML file
    run_reinvent(args.toml_file)

    # Read output from reinvent to dataframe
    df = pd.read_csv('data/sampling.csv')

    # Conditionally run the docking pipeline
    if args.dock:
        run_docking_pipeline()
    else:
        df.to_csv('data/dataframe.csv')


if __name__ == "__main__":
    main()
