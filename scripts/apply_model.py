from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_events
import matplotlib.pyplot as plt
import torch
import seisbench.models as sbm
from core.utils import pick_dict


# Load model
models = [torch.load("models/induced_stead.pth", map_location=torch.device('cpu')),
          sbm.PhaseNet.from_pretrained("stead")]

# Apply model to real data
cat = read_events("/scratch/gpi/seis/jheuel/ai_datasets/forge/catalogs/forge02.xml")
client = Client("IRIS")

# Loop over each model
for model in models:
    # Loop over each event and get picks
    for event in cat.events:
        picks, _ = pick_dict(event.picks)
        for network_station in picks.keys():
            for phase, time in picks[network_station].items():
                starttime = time
                break

            t = UTCDateTime(starttime)
            stream = client.get_waveforms(network=network_station.split(".")[0],
                                          station=network_station.split(".")[1],
                                          location="*", channel="?H?", starttime=t-30, endtime=t+50)

            annotations = model.annotate(stream)

            fig = plt.figure(figsize=(15, 10))
            axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})

            offset = annotations[0].stats.starttime - stream[0].stats.starttime
            for i in range(3):
                axs[0].plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
                if annotations[i].stats.channel[-1] != "N":  # Do not plot noise curve
                    axs[1].plot(annotations[i].times() + offset, annotations[i].data, label=annotations[i].stats.channel)

            axs[0].legend()
            axs[1].legend()
            plt.show()
