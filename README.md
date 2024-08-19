# rips-relay
RIPS 2024 Relay Team

## Pipeline for generating analogs and docking

### Usage

To use, simply run `python3 generate_analogs.py` in the terminal. You must be at the top of the rips-relay directory.

#### Arguments

* `--model` Specify which model to use. Choices are `reinvent`, `crem`, `coati`, or `safe`
* `--input_frag` Provides an option to include input fragment as a SMILES string. If `--pdb` is specified, this will be overwritten by the corresponding reference ligand.
* `--lead` Provides an option to include a lead molecule for which to compute a docking score. This will be included at the top of the output dataframe in `dataframe.csv`
* `--sample` Take a sample size from the generated distribution. If the sample size is larger than the size of the distribution, then the entire distribution will be returned
* `--remove_odd_rings` Flag to pass generated analogs through the `remove_odd_rings` filter
* `--dock` If you include this argument, the pipeline will attempt to dock the analogs to the protein in the `data` folder. `--pdb` must be specified in this case
* `--pdb` Enter the prefix of the pdb/sdf file for the protein-ligand pair



## Experiments

Included are various python notebooks detailing different experiments. We can include descriptions here

### [Visualizations](experiments/visualizations.ipynb)

A first look at the distributions of generated analogs using the Morgan fingerprint representation and PCA.

### [Fragment-Lead Pairs](experiments/fragment_lead_pairs.ipynb)

In this notebook, we attempt to investigate the relationship between the analogs generated from a given hit molecule and the corresponding lead.

### [CReM vs. REINVENT: A first look](experiments/crem_vs_reinvent.ipynb)

A first glance at the differences between CReM and REINVENT.

### [Visualizations 2](experiments/visualizations_2.ipynb)

A first look at the differences between all four models (i.e., REINVENT, CReM, COATI, and SAFE)

### [Piggyback](experiments/piggyback.ipynb)

Exploring a model's ability to rediscover leads via "piggybacking"

### [Feature Selection w/ Random Forest](experiments/feature_selection.ipynb)

In this experiment, we generated distributions of molecules from each model (REINVENT, CReM, COATI, SAFE). We then trained a random forest to classify the model from which a molecule was generated based on certain metrics from MolScore. Then, we were able to see which features had the highest importance (as determined by the RF), effectively indicating which metrics vary the most across models.

### [Characterizing Distributions](experiments/characterizing_distributions.ipynb)
