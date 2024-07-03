# rips-relay
RIPS 2024 Relay Team


### Usage

To run, simply run `python3 generate_analogs.py` in the terminal. 

The arguments are

* `--toml_file` followed by the path of the toml file. This specifies the path of the file used to configure REINVENT4.
* `--dock` If you add this argument, the pipeline will attempt to dock the analogs to the protein in the `data` folder.
* `--num_molecules` This requires further development. Default is 1. You can specify the number of initial molecules from which to generate analogs.
* `--remove_odd_rings` Default is True. Set to false if you want to keep molecules with odd ring systems.
* `--generate_ifp` Default is False. Specify whether or not you would like the interaction fingerprint stores as a csv file.
