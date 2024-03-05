import obspy.clients.fdsn.header
from obspy.clients.fdsn import Client
from obspy import read_events
import matplotlib.pyplot as plt
import torch
import seisbench.models as sbm
from core.utils import get_picks, phase_color

# Load model
models = [torch.load("models/induced_stead.pth", map_location=torch.device('cpu')),
          torch.load("models/induced_scratch.pth", map_location=torch.device('cpu')),
          torch.load("models/induced_original_sigma10.pth", map_location=torch.device('cpu')),
          torch.load("models/induced_scratch_sigma10.pth", map_location=torch.device('cpu')),
          torch.load("models/induced_scratch_sigma15.pth", map_location=torch.device('cpu')),
          torch.load("models/induced_stead_sigma10.pth", map_location=torch.device('cpu')),
          sbm.PhaseNet.from_pretrained("stead"),
          sbm.PhaseNet.from_pretrained("instance"),
          sbm.PhaseNet.from_pretrained("original"),
          sbm.PhaseNet.from_pretrained("scedc")]
model_names = ["Induced\nSTEAD", "scratch", "S10-orig", "S10", "S15", "S10-STEAD", "STEAD", "INSTANCE", "Original", "SCEDC"]

# Read catalog and load client to apply models to real data
cat = read_events("/scratch/gpi/seis/jheuel/ai_datasets/forge/catalogs/forge02.xml")
client = Client("IRIS")

# Get data for each event in catalog
for event in cat.events:
    # picks, status = pick_dict(event.picks)    # When picks and phase_hint are available
    picks, status = get_picks(event)  # When phases are only available in arrivals
    # Loop over each station in picks and get stream
    for station_network, phases in picks.items():
        # Get starttime for stream
        starttime = list(phases.values())[0]

        # Read waveforms from client
        try:
            stream = client.get_waveforms(network=station_network.split(".")[0],
                                          station=station_network.split(".")[1],
                                          location="*",
                                          channel="EH?",
                                          starttime=starttime - 30, endtime=starttime + 45)
            stream.sort(keys=["channel"], reverse=True)   # Sort stream to ZNE or Z12
            stream.detrend("demean")
            # # Add white noise
            # for trace in stream:
            #     noise = np.random.normal(loc=0, scale=np.std(trace.data), size=trace.stats.npts)
            #     trace.data = trace.data + np.random.uniform(2, 5) * noise
            print(stream)
        except obspy.clients.fdsn.header.FDSNNoDataException:
            continue

        # Loop over each model and annotate stream
        pn_predictions = []
        for model in models:
            pn_predictions.append(model.annotate(stream))

        # Create plot
        fig = plt.figure()    # Num subplots (waveforms + num of models)
        nrows = int(len(stream) + len(models))
        # time_trace = np.arange(stream[0].stats.npts) / stream[0].stats.sampling_rate
        # 1. Plot waveforms
        for subplotindex, trace in enumerate(stream):
            if subplotindex == 0:
                ax = fig.add_subplot(nrows, 1, subplotindex + 1)
            else:
                ax = fig.add_subplot(nrows, 1, subplotindex + 1, sharex=ax)
            ax.plot(trace.times(), trace.data,
                    color="k", label=trace.stats.channel)                           # Waveform
            for phase, time in phases.items():                                      # Phases from catalogue
                phase_sec = time - trace.stats.starttime
                ax.axvline(phase_sec, color=phase_color(phase), lw=1.5, zorder=0, label=phase)
            ax.legend()

        # 2. Plot predictions of PhaseNet
        for name, prediction in zip(model_names, pn_predictions):
            subplotindex += 1   # Subplotindex
            ax_pn = fig.add_subplot(nrows, 1, subplotindex + 1, sharex=ax)
            offset = prediction[0].stats.starttime - stream[0].stats.starttime
            for i in range(3):
                if prediction[i].stats.channel[-1] != "N":  # Do not plot noise curve
                    ax_pn.plot(prediction[i].times() + offset, prediction[i].data,
                               label=prediction[i].stats.channel)
                    ax_pn.legend()

            for phase, time in phases.items():                                      # Phases from catalogue
                phase_sec = time - stream[0].stats.starttime
                ax_pn.axvline(phase_sec, color=phase_color(phase), lw=0.7, zorder=0, alpha=0.6)

            # Add labels and set limits
            ax_pn.set_ylim([-0.05, 1.05])
            ax_pn.set_ylabel(name)

        # Set title and labels
        plt.xlabel(f"Time (s) since {stream[0].stats.starttime}")
        plt.suptitle(f"{stream[0].stats.network}.{stream[0].stats.station}.{stream[0].stats.location}")
        plt.subplots_adjust(hspace=0, top=0.94, bottom=0.06)
        plt.show()
