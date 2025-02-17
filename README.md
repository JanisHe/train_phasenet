# Training PhaseNet in SeisBench

### Requirements
Follow the instructions to install SeisBench on https://github.com/seisbench/seisbench.
Further you need to install
 - pyyaml
 - tqdm

### Training a model
To train a new model, download or create your own data set in SeisBench data format. 
Please follow the instructions and tutorials for downloading or creating SeisBench 
datasets on https://github.com/seisbench/seisbench.

To train an own PhaseNet model, modify the parameter file pn_parfile.yml and start training
by running python -u core/run_pn_parfile.py ./pn_parfile.yml