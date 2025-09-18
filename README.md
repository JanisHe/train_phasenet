# Training PhaseNet with SeisBench

### Requirements
Follow the instructions to install [SeisBench](https://github.com/seisbench/seisbench).
Further you need to install
 - `pyyaml`
 - `tqdm`

### Training a model
To train a new model, download or create your own data set in SeisBench data format. 
Please follow the instructions and tutorials for downloading or creating SeisBench 
datasets on https://github.com/seisbench/seisbench.

To train an own PhaseNet model, modify the parameter file `pn_parfile.yml` and start training
by running `python -u core/run_pn_parfile.py ./pn_parfile.yml`

### Applying your trained model
Once you have successfully trained your own PhaseNet model, you can apply your model to unseen
seismic data:
```
import seisbench.models as sbm
from obspy import read

model = sbm.PhaseNet.load("path/to/my/PhaseNet/model")  # Load your PhaseNet model
stream = obspy.read()  # Load your seismic data
picks = model.classify(stream,
                       batch_size=256,
                       P_threshold=0.2,
                       S_threshold=0.2,
                       blinding=[250, 250],
                       overlap=0.75,
                       stacking="max")
print(picks.picks)
```
Now, you should have a list of picks. For details about the arguments of the method `classify`, 
please read the [SeisBench docs](https://seisbench.readthedocs.io/en/stable/index.html).