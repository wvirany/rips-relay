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
import MDAnalysis as mda
import prolif as plf

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, CanonSmiles, Draw, MolFromSmiles, PandasTools
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
import useful_rdkit_utils as uru
from rdkit.Chem.rdFMCS import FindMCS

from crem.crem import mutate_mol

import safe as sf
import datamol as dm

from coati.generative.coati_purifications import embed_smiles
from coati.models.simple_coati2.io import load_coati2

import torch, gc

'''
Initializes environment variables and random seed; filters warnings
'''
def setup_environment():

    gc.collect()
    torch.cuda.empty_cache()

    warnings.filterwarnings('ignore')

    # Configuring path variables for OpenEye software
    os.environ["PATH"] += ":/usr/local/openeye/bin"
    os.environ["OE_LICENSE"] = "/home/fts_g_ucla_edu/Projects/oe_license.txt"

    # Setting random seed
    random.seed(42)


'''
Writes molecules to mol2mol.smi file for input to REINVENT4

Inputs:

- num_mols (default = 1): Number of molecules to be read from fragments.smi and written to mol2mol.smi;
                          these are then used as initial fragments from which to generate analogs
- smiles (default = False): If a smiles string is provided at runtime, this is used to generate analogs
                            instead of reading from fragments.smi

Returns: None
'''
def write_molecules(smiles):

    # Writes input smiles from command line argument to mol2mol.smi
    mol2mol = open("experiments/data/mol2mol.smi", "w")
    mol2mol.write(smiles)
    mol2mol.close()

    return smiles
    

'''
Removes odd ring systems from generated molecules

Input: Dataframe containing all generated molecules with columns SMILES, Input SMILES, Tanimoto, and NLL

Returns: Dataframe containing all filtered molecules with columns; SMILES, Input SMILES, Tanimoto
'''
def remove_odd_rings(df):
    ring_system_lookup = uru.RingSystemLookup.default()
    df['ring_systems'] = df.SMILES.apply(ring_system_lookup.process_smiles)
    df[['min_ring','min_freq']] = df.ring_systems.apply(uru.get_min_ring_frequency).to_list()
    df = df.query('min_freq > 100').copy()
    return df.iloc[:,0:4]


def gen_mol(encoder, tokenizer, smiles, coati_version=1, num_variations=100, noise_scale=0.15):

    # Embed the SMILES string
    smiles = CanonSmiles(smiles)

    vector = embed_smiles(smiles, encoder, tokenizer)

    # Noise is added as an isotropic Gaussian with std=noise_scale
   
    nearby_smiles = encoder.hcoati_to_2d_batch(
        h_coati=vector.unsqueeze(0).repeat(num_variations, 1),
        tokenizer=tokenizer,
        noise_scale=noise_scale,
    )

    # Retrieve canonical SMILES of generated analogs
    unique_valid_smiles = list(set([CanonSmiles(smi) for smi in nearby_smiles if MolFromSmiles(smi)]))

    # Store true if original molecule is in the set of generated analogs
    had_orig = smiles in unique_valid_smiles

    unique_valid_smiles = list(set([smiles] + unique_valid_smiles))

    # Generate molecular fingerprints
    fp = RDKFingerprint(MolFromSmiles(smiles), minPath=1, maxPath=7, fpSize=2048)
    fps = [RDKFingerprint(MolFromSmiles(x), minPath=1, maxPath=7, fpSize=2048) for x in unique_valid_smiles]

    # Compute tanimoto similarities between distributions and store as list of strings
    sim = BulkTanimotoSimilarity(fp, fps)
    sim_str = [str(round(x, 2)) for x in sim]

    unique_valid_smiles, sim_str = zip(*sorted(zip(unique_valid_smiles, sim_str), key=lambda x:x[1], reverse=True))

    if not had_orig:
        unique_valid_smiles, sim_str = zip(*[[i, f"{j} (Added)"] if i==smiles else [i, j] for i, j in zip(unique_valid_smiles, sim_str)])

    # Output for molecule generation
    print (f"Attempted {num_variations} COATI{coati_version} generations with a noise scale of {noise_scale} and generated {len(unique_valid_smiles)} unique structures.")
        
    return unique_valid_smiles


'''
Runs the REINVENT4 software to generate analogs

Input: .toml file with configuration settings

Returns: None
'''
def run_reinvent(input_frag):

    # Write .smi file with entry molecules
    write_molecules(input_frag)

    priors = ['high_similarity',
              'medium_similarity',
              'mmp',
              'scaffold_generic',
              'scaffold',
              'similarity']
    
    for prior in priors:
        # Assuming the reinvent command is installed and accessible
        command = f"reinvent priors/{prior}.toml --seed 42"
        subprocess.run(command, shell=True)

    # Read output from reinvent to dataframe
    df = pd.DataFrame()

    for prior in priors:
        
        file_path = f'experiments/data/analogs/sampling_{prior}.csv'
        temp_df = pd.read_csv(file_path)
        temp_df['Prior'] = prior

        df = pd.concat((df, temp_df))
    
    df.drop(['NLL'], axis=1, inplace=True)

    return df

def run_crem(initial):

    initial_mol = Chem.MolFromSmiles(initial)

    crem_db = 'crem_db/crem_db2.5.db'

    out_list = []
    mutate_list = list(mutate_mol(initial_mol, db_name=crem_db, return_mol=False))

    for idx, analog in enumerate(mutate_list):
        out_list.append([analog])
    
    df = pd.DataFrame(out_list, columns=["SMILES"])

    df['Model'] = 'crem'

    return df

def run_coati(initial):

    encoder2, tokenizer2 = load_coati2(
        freeze=True,
        device=torch.device("cuda:0"),
        doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl"
    )

    coati_smiles = gen_mol(encoder2, tokenizer2, initial, coati_version = 2, num_variations = 1000, noise_scale = 0.5)

    df = pd.DataFrame()
    df['SMILES'] = coati_smiles
    df['Model'] = 'coati'

    return df


def run_safe(initial):
    
    initial_mol = dm.to_mol(initial)

    designer = sf.SAFEDesign.load_default(verbose=True)

    safe_smiles = designer.de_novo_generation(sanitize=True, n_samples_per_trial=500)

    df = pd.DataFrame()
    df['SMILES'] = safe_smiles
    df['Model'] = 'safe'

    return df


'''
Runs the docking pipeline for generated analogs

Inputs: 
- df: Dataframe containing generated analogs
- generate_ifp (default = False): Boolean variable indicating whether or not to write interaction fingerprint to csv file

Returns: Dataframe and interaction fingerprint
'''
def run_docking_pipeline(df, pdb, model):

    PDB_FILEPATH = f"experiments/data/docking/{pdb}.pdb"
    OEDU_FILEPATH = f"experiments/data/docking/{pdb}.oedu"
    REF_MOL_FILEPATH = f"experiments/data/docking/{pdb}_ligand.sdf"

    # Placeholder for docking pipeline execution
    print("\nRunning docking pipeline...\n")

    # Additional code for docking pipeline can be added here
    subprocess.run(["spruce", "-in", PDB_FILEPATH, "-out", OEDU_FILEPATH], check=True)

    df['Name'] = [f"MOL{i:04d}" for i in range(len(df))]
    df[["SMILES","Name"]].to_csv("experiments/data/docking/analogs.smi", sep=" ", header=None, index=False)

    # Run omega and hybrid software to simulate docking
    subprocess.run(["/usr/local/openeye/bin/omega2", "-in", "experiments/data/docking/analogs.smi", "-out", "experiments/data/docking/analogs.oeb", "-strictstereo", "false", "-log", "/dev/null"], check=True)
    subprocess.run(["/usr/local/openeye/bin/hybrid", "-receptor", OEDU_FILEPATH, "-dbase", "experiments/data/docking/analogs.oeb", "-out", "experiments/data/docking/docked.sdf"], check=True)

    docked_df = PandasTools.LoadSDF("experiments/data/docking/docked.sdf") # Loads SDF file to dataframe
    ref_mol = Chem.MolFromMolFile(REF_MOL_FILEPATH) # Reads reference ligand provided for binding site

    rmsd_list = [mcs_rmsd(ref_mol, x) for x in docked_df.ROMol.values]  # Computes the RMSD for maximum common substructure between reference ligand and generated analogs
    docked_df['rmsd'] = rmsd_list   # Stores RMSD between reference ligand and analogs in RMSD column of the dataframe

    # df_rmsd_ok = docked_df.query("rmsd <= 4").copy()    # Keep only analogs with less than 2 RMSD with respect to reference ligand

    # PandasTools.WriteSDF(docked_df,"experiments/data/docking/docked_df.sdf")

    # Generate new columns in dataframe

    mol_ids = docked_df['ID'].to_list()  # Molecular IDS of analogs which docked successfully

    # df['Success'] = False   # Initialize new column in dataframe indicating successful docking

    # # Iterate over rows of dataframe of ALL analogs. If molecular ID is present in list of those which docked successfully, set SUCCESS = True for that molecule
    # for index, row in df.iterrows():
    #     if row['Name'] in mol_ids:
    #         df['Success'][index] = True

    # Store docking scores in original dataframe and rename column to 'Docking score'
    df = df.merge(docked_df[['ID', 'HYBRID Chemgauss4 score', 'rmsd']], left_on='Name', right_on='ID', how='left')
    df.rename(columns={'HYBRID Chemgauss4 score' : 'Docking score'}, inplace=True)

    df.drop(['Name'], axis=1, inplace=True)

    SDF_FILEPATH = "experiments/data/docking/docked.sdf"

    fp = plf.Fingerprint()

    mol = Chem.MolFromPDBFile(PDB_FILEPATH, removeHs=False)
    prot = plf.Molecule(mol)
    suppl = plf.sdf_supplier(SDF_FILEPATH)
    fp.run_from_iterable(suppl, prot, progress=True)
    df_ifp = fp.to_dataframe()
    df_ifp.columns = df_ifp.columns.droplevel(0)
    df_ifp['ID'] = mol_ids


    IFP_FILEPATH = f'experiments/data/{model}_ifp.csv'

    df_ifp.to_csv(IFP_FILEPATH)

    return df


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


def main():
    parser = argparse.ArgumentParser(description="Run reinvent and optionally the docking pipeline")
    parser.add_argument("--model", type=str, choices=['reinvent', 'crem', 'coati', 'safe'], default='reinvent', help="Specify which model to use")
    parser.add_argument('--input_frag', nargs='?', const=False, type=str, help='Enter a smiles string for input')
    parser.add_argument('--lead', nargs='?', const=False, type=str, help='Enter a corresponding lead for which to compare docking score in fragment-lead-pair-style experiments')
    parser.add_argument("--sample", nargs="?", const=False, type=int, help='Enter a number for size of random sample of generated molecules')
    parser.add_argument("--remove_odd_rings", action="store_true", help="Flag to filter out molecules containing odd ring systems")
    parser.add_argument("--dock", action="store_true", help="Flag to run the docking pipeline after reinvent")
    parser.add_argument("--pdb", nargs="?", const=False, type=str, help='Enter the prefix of a .pdb/.sdf file')

    args = parser.parse_args()

    # Setup the environment
    setup_environment()

    initial = False

    if args.input_frag:
        initial = args.input_frag

    if args.pdb:
        initial_mol = Chem.MolFromMolFile(f"experiments/data/docking/{args.pdb}_ligand.sdf")
        initial = Chem.MolToSmiles(initial_mol)


    if args.model=='reinvent':
        df = run_reinvent(initial)
    elif args.model=='crem':
        df = run_crem(initial)
    elif args.model=='coati':
        df = run_coati(initial)
    elif args.model=='safe':
        df = run_safe(initial)

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)


    if args.lead:
        df.loc[-1] = [args.lead, args.input_frag, None, None]  # adding a row
        df.index = df.index + 1  # shifting index
        df.sort_index(inplace=True)

    if args.remove_odd_rings:
        # Remove odd ring systems
        df = remove_odd_rings(df)

    # Conditionally run the docking pipeline
    if args.dock:
        df = run_docking_pipeline(df, args.pdb, args.model)
        
    DF_FILEPATH = f'experiments/data/{args.model}_dataframe.csv'

    df.to_csv(DF_FILEPATH)


if __name__ == "__main__":
    main()
