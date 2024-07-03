# rips-relay
RIPS 2024 Relay Team

## Pipeline for generating analogs and docking

### Usage

To use, simply run `python3 generate_analogs.py` in the terminal. 

#### Arguments

* `--toml_file` followed by the path of the toml file. This specifies the path of the file used to configure REINVENT4.
* `--dock` If you include this argument, the pipeline will attempt to dock the analogs to the protein in the `data` folder.
* `--num_molecules` This requires further development. Default is 1. You can specify the number of initial molecules from which to generate analogs. They are read from `fragments.smi`.
* `--input_frag` Provides an option to include input fragment as a SMILES string as a command line argument at runtime. If so, the pipeline does not read molecules from `fragments.smi`.
* `--lead` Provides an option to include a lead molecules for which to compute a docking score. This will be included at the top of the output dataframe in `dataframe.csv`.
* `--remove_odd_rings` Default is True. Set to false if you want to keep molecules with odd ring systems.
* `--generate_ifp` Default is False. Specify whether or not you would like the interaction fingerprint stores as a csv file.



## Experiments

Included are various python notebooks detailing different experiments. We can include descriptions here

### [Visualizations](experiments/visualizations.ipynb)

A first look at the distributions of generated analogs using the Morgan fingerprint representation and PCA.

### [Fragment-Lead Pairs](experiments/fragment_lead_pairs.ipynb)

In this notebook, we attempt to investigate the relationship between the analogs generated from a given hit molecule and the corresponding lead.
