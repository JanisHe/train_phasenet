import os
import shutil
import subprocess
import pathlib
import warnings
import torch
import logging
import time
import logging
import yaml
import requests
import contextlib
import seisbench
import obspy
import socket
import contextlib

import numpy as np
import pandas as pd

import datetime as dt
import seisbench.data as sbd
import seisbench.models as sbm
import seisbench.generate as sbg
from tqdm import tqdm
from mpi4py import MPI
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from typing import Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding
from tqdm.auto import tqdm
import torch.utils.data.distributed as datadist


log = logging.getLogger("propulate")  # Get logger instance.
SUBGROUP_COMM_METHOD = "nccl-slurm"
GPUS_PER_NODE = 4


class Metrics:
    """

    """
    def __init__(self, probabilities, residuals, predictions=None,
                 true_pick_prob=0.5, arrival_residual=10):
        self.probabilities = probabilities
        self.residuals = residuals
        self.true_pick_prob = true_pick_prob
        self.arrival_residual = arrival_residual
        self.predictions = predictions

        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

        if self.predictions is not None:
            self.true_false_positives()

    def __str__(self):
        pass

    def true_false_positives(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        for prediction, probability, residual in zip(self.predictions, self.probabilities, self.residuals):
            if not is_nan(prediction):
                if probability >= self.true_pick_prob and abs(residual) <= self.arrival_residual:
                    self.true_positive += 1
                elif probability >= self.true_pick_prob and abs(residual) > self.arrival_residual:
                    self.false_positive += 1
                elif probability < self.true_pick_prob or abs(residual) > self.arrival_residual:
                    self.false_negative += 1

    @property
    def precision(self, eps=1e-6) -> float:
        return self.true_positive / (self.true_positive + self.false_positive + eps)

    @property
    def recall(self, eps=1e-6) -> float:
        return self.true_positive / (self.true_positive + self.false_negative + eps)

    @property
    def f1_score(self, eps=1e-6) -> float:
        return 2 * ((self.precision * self.recall) / (
                    self.precision + self.recall + eps))


class VectorCrossEntropyLoss:
    """
    Vector cross entropy as definded in Zhu & Beroza (2018).

    H(p, q) = \sum_{i=-1}^{3} \sum_x p_i(x) * log(q_i(x))
    p: true probabiliyt distribution
    q: predicted distribution
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true, eps=1e-10):
        """
        :param y_pred:
        :param y_true:
        :param eps:
        """
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(-1)                 # Mean along sample dimension and sum along pick dimension
        h = h.mean()                           # Mean over batch axis

        return -h


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """

        """
        mse = (y_true - y_pred) ** 2
        mse = mse.mean(-1).sum(-1)
        mse = mse.mean()

        return mse


def is_nan(num):
    return num != num


def rms(x):
    """
    Root-mean-square of array x
    :param x:
    :return:
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal, noise, decibel=True):
    """
    SNR in dB
    :param signal:
    :param noise:
    :param decibel:
    :return:
    """
    if decibel is True:
        return 20 * np.log10(rms(signal) / rms(noise))
    else:
        return rms(signal) / rms(noise)


def snr(signal, noise, decibel=True):
    """
    Wrapper for signal-to-noise ratio
    """
    return signal_to_noise_ratio(signal=signal, noise=noise, decibel=decibel)


def snr_pick(trace: obspy.Trace,
             picktime: obspy.UTCDateTime,
             window=5,
             **kwargs):
    """
    Computes SNR with a certain time window around a pick
    """
    pick_sample = int((picktime - trace.stats.starttime) * trace.stats.sampling_rate)
    window_len = int(window * trace.stats.sampling_rate)

    if pick_sample - window_len < 0:
        noise_win_begin = 0
    else:
        noise_win_begin = pick_sample - window_len

    return snr(signal=trace.data[pick_sample:pick_sample + window_len],
               noise=trace.data[noise_win_begin:pick_sample],
               **kwargs)


def pick_dict(picks):
    """
    Create dictionary for each station that contains P and S phases

    returns: dict
             keys: network_code.station_code
             values: dict with phase_hints and time of pick
    """
    station_pick_dct = {}
    station_status_dct = {}
    for pick in picks:
        station_id = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
        if station_id in station_pick_dct.keys():
            station_pick_dct[station_id].update({pick["phase_hint"]: pick["time"]})
            station_status_dct[station_id].update({f"{pick['phase_hint']}_status": pick.evaluation_mode})
        else:
            station_pick_dct.update({station_id: {pick["phase_hint"]: pick["time"]}})
            station_status_dct.update({station_id: {f"{pick['phase_hint']}_status": pick.evaluation_mode}})

    return station_pick_dct, station_status_dct


def get_picks(event):
    """
    Function creates a dictionary with all picks from 'event.origins[0].arrivals' and 'event.picks'.
    The keys of the returned dictionary are named 'network.station' and contains a further dictionary with
    the phase hints and UTCDateTimes of the different phases.

    Works only for FORGE events
    """
    # Test if events has arrivals and picks
    if len(event.origins[0].arrivals) == 0:
        msg = "Event does not have arrivals."
        raise ValueError(msg)

    if len(event.picks) == 0:
        msg = "Event does not have picks."
        raise ValueError(msg)

    # Create list of all pick resource_ids
    pick_rid = [pick.resource_id.id.split("/")[-1] for pick in event.picks]

    # Loop over each arrival
    for arrival in event.origins[0].arrivals:
        # Find resource_id of arrival in pick_rid
        try:
            pick_index = pick_rid.index(arrival.resource_id.id.split("/")[-1])
            # Add phase to picks
            event.picks[pick_index].phase_hint = arrival.phase
        except ValueError:
            pass

    return pick_dict(event.picks)


def sort_event_list(event_list):
    """
    Sort event list by dates, from early to late
    :param event_list:
    :return:
    """
    # Create list with all dates
    dates = [event.origins[0].time.datetime for event in event_list]

    # Sort event_list by dates
    sorted_events = [x for _, x in zip(sorted(dates), event_list)]

    return sorted_events


def merge_catalogs(catalogs: list, sort: bool = True) -> obspy.Catalog:
    all_events = []
    for catalog in catalogs:
        for event in catalog.events:
            all_events.append(event)

    # Sort events by date
    if sort is True:
        all_events = sort_event_list(event_list=all_events)

    return obspy.Catalog(events=all_events)


def normalize(array: np.array):
    """
    Removing mean from array and dividing by its standard deviation.
    :param array: numpy array

    :returns: normalized array
    """
    return (array - np.mean(array)) / np.std(array)


def phase_color(phase):
    if phase == "P":
        return "b"
    elif phase == "S":
        return "r"
    else:
        raise Exception


def check_parameters(parameters: dict) -> dict:
    """
    Checks parameters from .yml file and corrects if necessary
    """
    # Modify datasets if tmpdir in parameters
    if parameters.get("tmpdir"):
        tmpdir = subprocess.check_output("cd $TMPDIR && pwd", shell=True)
        tmpdir = tmpdir.decode("utf-8")
        tmpdir = tmpdir.replace("\n", "")

        # Modify datasets in parameters
        for dname, pathname in parameters["datasets"][0].items():
            parameters["datasets"][0][dname] = os.path.join(tmpdir, pathlib.Path(pathname).parts[-1])

    return parameters


def filter_dataset(filter: dict, dataset: sbd.WaveformDataset):
    """
    Filtering metadata of seisbench dataset. Since filtering is inplace, nothing is returned.
    keywords of filter:
    - operation: "<" or ">"
    - item: which column in metadata
    - threshold: value for threshold
    """
    if filter["operation"] == "<":
        mask = dataset.metadata[filter["item"]] < filter["threshold"]
    elif filter["operation"] == ">":
        mask = dataset.metadata[filter["item"]] > filter["threshold"]
    else:
        msg = f'Filter operation {filter["operation"]} is not known'
        raise ValueError(msg)

    dataset.filter(mask, inplace=True)


def read_datasets(parameters: dict,
                  component_order: str = "ZNE",
                  dataset_key: str = "datasets",
                  filter: (dict, None) = None,):
    """
    Read seisbench dataset from parameter file.
    """
    for lst in parameters[dataset_key]:
        for dataset_count, dataset in enumerate(lst.values()):
            if dataset_count == 0:
                sb_dataset = sbd.WaveformDataset(path=pathlib.Path(dataset),
                                                 sampling_rate=parameters["sampling_rate"],
                                                 component_order=component_order)
                if filter:
                    filter_dataset(filter=filter, dataset=sb_dataset)
            else:
                subset = sbd.WaveformDataset(path=pathlib.Path(dataset),
                                             sampling_rate=parameters["sampling_rate"],
                                             component_order=component_order)
                if filter:
                    filter_dataset(filter=filter, dataset=subset)
                sb_dataset += subset

    return sb_dataset


def add_fake_events_to_metadata(sb_dataset: sbd.WaveformDataset,
                                number: int):
    """
    Copying metadata to add fake events
    """
    # Convert metadata to dictionary
    metadata_dct = sb_dataset.metadata.to_dict(orient="list")
    for i in range(number):
        rand_data_index = np.random.randint(0, len(sb_dataset.metadata))
        for key in metadata_dct:
            metadata_dct[key].append(metadata_dct[key][rand_data_index])
    metadata = pd.DataFrame(metadata_dct)
    sb_dataset._metadata = metadata




def get_phase_dict():
    map_phases = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_TPg_arrival_sample": "P",
        "trace_TSg_arrival_sample": "P",
        "trace_APg_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",
        "trace_ASg_arrival_sample": "S",
    }

    return map_phases


def inverse_dict(dictionary):
    """
    Swap keys and values. If values are not unique, a list is used instead

    :param dictionary:
    :return:
    """
    # Get unique values form map_phases
    uniques = set(list(dictionary.values()))
    inv_dict = {key: [] for key in uniques}

    # Loop over all items in dictionary
    for key, value in dictionary.items():
        inv_dict[value].append(key)

    return inv_dict


def map_arrivals(dataframe: pd.DataFrame,
                 map_phases: dict = None) -> pd.DataFrame:
    """

    df = pd.read_csv("/home/jheuel/scratch/ai_datasets/floodrisk/metadata.csv")
    map_phases = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_TPg_arrival_sample": "P",
        "trace_TSg_arrival_sample": "P",
        "trace_APg_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",
        "trace_ASg_arrival_sample": "S",
    }

    :param dataframe:
    :param map_phases:
    :return:
    """
    # Default dictionary for mapping phases
    if not map_phases:
        map_phases = get_phase_dict()

    # Get unique values form map_phases
    inv_phases = inverse_dict(map_phases)

    # Loop over each row in pandas dataframe and create new columns
    new_columns = {key: [] for key in inv_phases.keys()}
    for index in dataframe.index:
        for key in inv_phases.keys():   # P | S
            for df_keys_index, value in enumerate(inv_phases[key]):
                if value in dataframe.keys():
                    if np.isnan(dataframe[value][index]) == False:
                        new_columns[key].append(dataframe[value][index])
                        break
            else:
                # Add nan if phase does not exist
                new_columns[key].append(np.nan)

    # Add new columns to dataframe
    for key in inv_phases:
        dataframe[key] = pd.Series(new_columns[key], index=dataframe.index)

    return dataframe


def get_sb_phase_value(phase: str, phasennet_model: seisbench.models.phasenet.PhaseNet) -> int:
    try:
        return phasennet_model.labels.index(phase)
    except ValueError:
        ValueError(f"Phase {phase} is not in model labels ({phasennet_model.labels}).")


def get_true_pick(batch: dict, index: int, phasenet_model: seisbench.models.phasenet.PhaseNet,
                  phase: str = "P") -> Union[str, None]:
    phase_index = get_sb_phase_value(phase=phase, phasennet_model=phasenet_model)
    true_pick_index = np.argmax(batch["y"][index, phase_index, :].detach().numpy())

    if true_pick_index == 0:
        return None
    else:
        return true_pick_index


def get_predicted_pick(prediction: torch.Tensor, index: int, true_pick: (int, None),
                       phasenet_model: seisbench.models.phasenet.PhaseNet,
                       sigma=30, phase: str = "P", win_len_factor=10) -> Union[int, None]:
    # Return None if true_pick is None
    if not true_pick:
        return None

    # Get phase index
    phase_index = get_sb_phase_value(phase=phase, phasennet_model=phasenet_model)

    # If GPU is used, convert prediction from cuda to numpy
    prediction = prediction.cpu()

    # Find maximum of predicted pick
    lower_bound = true_pick - int(win_len_factor * sigma)
    upper_bound = true_pick + int(win_len_factor * sigma)
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > prediction.shape[2]:
        upper_bound = prediction.shape[2]
    pred_pick_index = np.argmax(prediction[index, phase_index, lower_bound:upper_bound].detach().numpy())

    # Add offset to pred_pick_index
    pred_pick_index += true_pick - int(win_len_factor * sigma)

    if pred_pick_index == 0:
        return None
    else:
        return pred_pick_index


def get_pick_probabilities(prediction_sample: (int, None), prediction: torch.Tensor, index: int,
                           phasenet_model: seisbench.models.phasenet.PhaseNet, phase: str = "P") -> Union[float, None]:

    phase_index = get_sb_phase_value(phase=phase, phasennet_model=phasenet_model)
    if prediction_sample:
        return prediction[index, phase_index, prediction_sample].item()
    else:
        return None


def pick_residual(true: int, predicted: int) -> (int, np.nan):
    if not true or not predicted:
        return np.nan
    else:
        return predicted - true


def get_picks(model, dataloader, sigma=30, win_len_factor=10, return_data=False):
    pick_results = {"true_P": np.empty(len(dataloader.dataset)),
                    "true_S": np.empty(len(dataloader.dataset)),
                    "pred_P": np.empty(len(dataloader.dataset)),
                    "pred_S": np.empty(len(dataloader.dataset)),
                    "prob_P": np.empty(len(dataloader.dataset)),
                    "prob_S": np.empty(len(dataloader.dataset)),
                    "residual_P": np.empty(len(dataloader.dataset)),
                    "residual_S": np.empty(len(dataloader.dataset))}

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            pred = model(batch["X"].to(model.device))
            for index in range(pred.shape[0]):
                # Find true P and S phase arrival
                true_p_samp = get_true_pick(batch=batch, index=index, phase="P", phasenet_model=model)
                true_s_samp = get_true_pick(batch=batch, index=index, phase="S", phasenet_model=model)

                # Find predicted P and S arrival
                pred_p_samp = get_predicted_pick(prediction=pred, index=index, true_pick=true_p_samp,
                                                 sigma=sigma, phase="P", win_len_factor=win_len_factor,
                                                 phasenet_model=model)
                pred_s_samp = get_predicted_pick(prediction=pred, index=index, true_pick=true_s_samp,
                                                 sigma=sigma, phase="S", win_len_factor=win_len_factor,
                                                 phasenet_model=model)

                # Get pick probabilities for P and S
                p_prob = get_pick_probabilities(prediction=pred, prediction_sample=pred_p_samp,
                                                index=index, phase="P", phasenet_model=model)
                s_prob = get_pick_probabilities(prediction=pred, prediction_sample=pred_s_samp,
                                                index=index, phase="S", phasenet_model=model)

                # Write results to dictionary
                pick_results["true_P"][index + (batch_index * dataloader.batch_size)] = true_p_samp
                pick_results["true_S"][index + (batch_index * dataloader.batch_size)] = true_s_samp
                pick_results["pred_P"][index + (batch_index * dataloader.batch_size)] = pred_p_samp
                pick_results["pred_S"][index + (batch_index * dataloader.batch_size)] = pred_s_samp
                pick_results["prob_P"][index + (batch_index * dataloader.batch_size)] = p_prob
                pick_results["prob_S"][index + (batch_index * dataloader.batch_size)] = s_prob
                pick_results["residual_P"][index + (batch_index * dataloader.batch_size)] = pick_residual(
                    true=true_p_samp, predicted=pred_p_samp)
                pick_results["residual_S"][index + (batch_index * dataloader.batch_size)] = pick_residual(
                    true=true_s_samp, predicted=pred_s_samp)

                if return_data is True:
                    # Allocate memory for data and predictions
                    if batch_index + index == 0:
                        pick_results.update({"data": np.empty(shape=(len(dataloader.dataset),
                                                                     batch["X"].shape[1],
                                                                     batch["X"].shape[2])),
                                             "prediction": np.empty(shape=(len(dataloader.dataset),
                                                                           batch["y"].shape[1],
                                                                           batch["y"].shape[2]))})
                    # Write data and predictions
                    pick_results["data"][index + (batch_index * dataloader.batch_size), :, :] = batch["X"][index]
                    pick_results["prediction"][index + (batch_index * dataloader.batch_size), :, :] = pred[index]

    return pick_results


def residual_histogram(residuals, axes, bins=60, xlim=(-100, 100)):
    counts, bins = np.histogram(residuals, bins=bins, range=xlim)
    axes.hist(bins[:-1], bins, weights=counts, edgecolor="b")

    return axes


def add_metrics(axes, metrics: Metrics):
    # TODO: Add mean and standard deviation
    textstr = (f"Precision: {np.round(metrics.precision, 2)}\n"
               f"Recall: {np.round(metrics.recall, 2)}\n"
               f"F1 score: {np.round(metrics.f1_score, 2)}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axes.text(0.05, 0.95, textstr, transform=axes.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)


def test_model(model: seisbench.models.phasenet.PhaseNet,
               test_dataset: seisbench.data.base.MultiWaveformDataset,
               plot_residual_histogram: bool = False,
               **parameters):
    """

    """
    test_generator = sbg.GenericGenerator(test_dataset)

    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()), samples_before=int(parameters["nsamples"] / 3),
                               windowlen=parameters["nsamples"],
                               selection="first", strategy="move"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=model.norm),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=get_phase_dict(), sigma=parameters["sigma"], dim=0,
                                  model_labels=model.labels, noise_column=True)
    ]
    test_generator.add_augmentations(augmentations)
    test_loader = DataLoader(dataset=test_generator, batch_size=parameters["batch_size"],
                             shuffle=False, num_workers=parameters["nworkers"],
                             worker_init_fn=worker_seeding, drop_last=False)

    picks_and_probs = get_picks(model=model, dataloader=test_loader, sigma=parameters["sigma"],
                                win_len_factor=parameters["win_len_factor"])

    # Evaluate metrics for P and S
    # 1. Determine true positives (tp), false positives (fp), and false negatives (fn) for P and S phases
    metrics_p = Metrics(probabilities=picks_and_probs["prob_P"], residuals=picks_and_probs["residual_P"],
                        true_pick_prob=parameters["true_pick_prob"], arrival_residual=parameters["arrival_residual"],
                        predictions=picks_and_probs["pred_P"])
    metrics_s = Metrics(probabilities=picks_and_probs["prob_S"], residuals=picks_and_probs["residual_S"],
                        true_pick_prob=parameters["true_pick_prob"], arrival_residual=parameters["arrival_residual"],
                        predictions=picks_and_probs["pred_S"])

    # 2. Plot time arrival residuals for P and S
    if plot_residual_histogram is True:
        # Get filename from parameters
        filename = parameters.get("filename")
        if not filename:
            filename = pathlib.Path(parameters["model_name"]).stem

        # Create directory if it does not exist
        if not os.path.exists(os.path.join(".", "metrics")):
            os.makedirs(os.path.join(".", "metrics"))

        # Plot figure
        fig = plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, sharey=ax1)
        residual_histogram(residuals=picks_and_probs["residual_P"] / parameters["sampling_rate"],
                           axes=ax1,
                           xlim=(-10 * parameters["sigma"] / parameters["sampling_rate"],
                                 10 * parameters["sigma"] / parameters["sampling_rate"]))
        residual_histogram(residuals=picks_and_probs["residual_S"] / parameters["sampling_rate"],
                           axes=ax2,
                           xlim=(-10 * parameters["sigma"] / parameters["sampling_rate"],
                                 10 * parameters["sigma"] / parameters["sampling_rate"]))
        add_metrics(ax1, metrics=metrics_p)
        add_metrics(ax2, metrics=metrics_s)
        ax1.set_title("P residual")
        ax2.set_title("S residual")
        ax1.set_xlabel("$t_{pred}$ - $t_{true}$ (s)")
        ax2.set_xlabel("$t_{pred}$ - $t_{true}$ (s)")
        ax1.set_ylabel("Counts")
        fig.savefig(fname=os.path.join(".", "metrics", f"{filename}_residuals.png"), dpi=250)

    return metrics_p, metrics_s


def torch_process_group_init(comm: MPI.Comm, method: str) -> None:
    """
    Create the torch process group.

    Parameters
    ----------
    comm : MPI.Comm
        Communciator used for training the model in data parallel fashion
    method : str
        The method to use to initialize the process group.
        Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
        If CUDA is not available, ``gloo`` is automatically chosen for the method.
    """
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    comm_rank, comm_size = comm.rank, comm.size

    # Get master address and port
    port = 29500

    if comm_size == 1:
        return

    master_address = f"{socket.gethostname()[:-7]}i"  # THIS IS THE NEW BIT! IT WILL PULL OUT THE rank-0 NODE NAME
    # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
    master_address = comm.bcast(str(master_address), root=0)

    # Save environment variables.
    os.environ["MASTER_ADDR"] = master_address
    # # OLD
    # master_address = os.environ["MASTER_ADDR"]
    # # Each rank needs to get the hostname of rank 0 of its group.
    # # master_address = comm.bcast(str(master_address), root=0)
    #
    # # Save environment variables.
    # # os.environ["MASTER_ADDR"] = master_address
    # Use the default PyTorch port.
    os.environ["MASTER_PORT"] = str(port)


    if not torch.cuda.is_available():
        method = "gloo"
        # log.info("No CUDA devices found: Falling back to gloo.")
        print("No CUDA devices found: Falling back to gloo.")
    else:
        # log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        num_cuda_devices = torch.cuda.device_count()
        device_number = MPI.COMM_WORLD.rank % num_cuda_devices
        # log.info(f"device count: {num_cuda_devices}, device number: {device_number}")
        print(f"device count: {num_cuda_devices}, device number: {device_number}")
        torch.cuda.set_device(device_number)

    time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
    if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
        dist.init_process_group(
            backend="nccl",
            rank=comm_rank,
            world_size=comm_size,
        )

    elif method == "nccl-slurm":  # Use NCCL with a TCP store.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    elif method == "gloo":  # Use gloo.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    else:
        raise NotImplementedError(
            f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]!"
        )

    # Call a barrier here in order for sharp to use the default comm.
    if dist.is_initialized():
        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == comm_size, "Failed test of dist!"
    else:
        disttest = None
    # log.info(
    #     f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}"
    # )
    print(f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}")


def torch_process_group_init_propulate(subgroup_comm: MPI.Comm, method: str) -> None:
    """
    Create the torch process group of each multi-rank worker from a subgroup of the MPI world.

    Parameters
    ----------
    subgroup_comm : MPI.Comm
        The split communicator for the multi-rank worker's subgroup. This is provided to the individual's loss function
        by the ``Islands`` class if there are multiple ranks per worker.
    method : str
        The method to use to initialize the process group.
        Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
        If CUDA is not available, ``gloo`` is automatically chosen for the method.
    """
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    comm_rank, comm_size = subgroup_comm.rank, subgroup_comm.size

    # Get master address and port
    # Don't want different groups to use the same port.
    subgroup_id = MPI.COMM_WORLD.rank // comm_size
    port = 29500 + subgroup_id

    if comm_size == 1:
        return
    master_address = os.environ["MASTER_ADDR"]
    # master_address = socket.gethostname()
    # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
    master_address = subgroup_comm.bcast(str(master_address), root=0)

    # Save environment variables.
    # os.environ["MASTER_ADDR"] = master_address
    # Use the default PyTorch port.
    os.environ["MASTER_PORT"] = str(port)

    if not torch.cuda.is_available():
        method = "gloo"
        log.info("No CUDA devices found: Falling back to gloo.")
    else:
        log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        num_cuda_devices = torch.cuda.device_count()
        device_number = MPI.COMM_WORLD.rank % num_cuda_devices
        log.info(f"device count: {num_cuda_devices}, device number: {device_number}")
        torch.cuda.set_device(device_number)

    time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
    if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
        dist.init_process_group(
            backend="nccl",
            rank=comm_rank,
            world_size=comm_size,
        )

    elif method == "nccl-slurm":  # Use NCCL with a TCP store.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    elif method == "gloo":  # Use gloo.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=60),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    else:
        raise NotImplementedError(
            f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]!"
        )

    # Call a barrier here in order for sharp to use the default comm.
    if dist.is_initialized():
        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == comm_size, "Failed test of dist!"
    else:
        disttest = None
    log.info(
        f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}"
    )


def get_data_loaders(comm: MPI.Comm, parameters, model):
    # Read waveform datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      component_order=model.component_order,
                                      dataset_key="datasets",
                                      filter=parameters.get("filter"))

    # Add fake events to metadata
    if parameters.get("add_fake_events"):
        add_fake_events_to_metadata(sb_dataset=seisbench_dataset,
                                    number=int(len(seisbench_dataset) * parameters["add_fake_events"] / 100))

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Define generators for training and validation
    train_generator = sbg.GenericGenerator(train)
    val_generator = sbg.GenericGenerator(validation)

    # Build augmentations and labels
    # Ensure that all phases are in requested window
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()),
                               samples_before=int(0.8 * parameters["nsamples"]),
                               windowlen=int(1.5 * parameters["nsamples"]),
                               selection="first",
                               strategy="variable"),
        sbg.RandomWindow(windowlen=parameters["nsamples"],
                         strategy="pad"),
        sbg.ProbabilisticLabeller(shape=parameters["labeler"],
                                  label_columns=get_phase_dict(), sigma=parameters["sigma"],
                                  dim=0, model_labels=model.labels, noise_column=True)
    ]

    if parameters.get("rotate") is True:
        augmentations.append(sbg.RotateHorizontalComponents())

    # Add RealNoise to augmentations if noise_datasets are in parmeters
    if parameters.get("noise_datasets"):
        noise_dataset = read_datasets(parameters=parameters, dataset_key="noise_datasets")
        # TODO: trace_Z_snr is hard coded
        augmentations.append(
            sbg.OneOf(
                augmentations=[sbg.RealNoise(
                    noise_dataset=noise_dataset,
                    metadata_thresholds={"trace_Z_snr_db": 10}
                )],
                probabilities=[0.5]
            )
        )

    # Change dtype of data (necessary for PyTorch and the last augmentation step)
    augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=model.norm))
    augmentations.append(sbg.ChangeDtype(np.float32))

    # Add augmentations to generators
    train_generator.add_augmentations(augmentations=augmentations)
    val_generator.add_augmentations(augmentations=augmentations)

    if comm.size > 1:  # Make the samplers use the torch world to distribute data
        train_sampler = datadist.DistributedSampler(train_generator,
                                                    seed=42)
        val_sampler = datadist.DistributedSampler(val_generator,
                                                  seed=42)
    else:
        train_sampler = None
        val_sampler = None

    num_workers = parameters["nworkers"]
    print(f"Use {num_workers} workers in dataloader.")

    # Define generators to load data
    train_loader = DataLoader(dataset=train_generator,
                              batch_size=parameters["batch_size"],
                              num_workers=parameters["nworkers"],
                              pin_memory=True,
                              persistent_workers=True,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler)

    val_loader = DataLoader(dataset=val_generator,
                            batch_size=parameters["batch_size"],
                            num_workers=parameters["nworkers"],
                            pin_memory=True,
                            persistent_workers=True,
                            shuffle=(val_sampler is None),
                            sampler=val_sampler)

    return train_loader, val_loader, test


def ind_loss(h_params: dict[str, int | float],
             subgroup_comm: MPI.Comm) \
        -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    h_params : dict[str, int | float]
        The hyperparameters to be optimized evolutionarily. Here, batch_size and learning_rate.
    subgroup_comm : MPI.Comm
        Each multi-rank worker's subgroup communicator.

    Returns
    -------
    float
        The trained model's validation loss.
    """
    torch_process_group_init_propulate(subgroup_comm, method=SUBGROUP_COMM_METHOD)

    # Extract hyperparameter combination to test from input dictionary.
    lr = h_params["learning_rate"]

    parfile = "../pn_parfile.yml"
    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    filename = pathlib.Path(parameters["model_name"]).stem
    parameters["filename"] = filename

    # Updating batch_size in parameters from hyperparameters
    parameters["batch_size"] = h_params["batch_size"]

    # Set number of workers for PyTorch
    # https://github.com/pytorch/pytorch/issues/101850
    os.sched_setaffinity(0, range(os.cpu_count()))

    # Make copy of parfile and rename it by filename given in parameters
    if not os.path.exists("./parfiles"):
        os.makedirs("./parfiles")
    try:
        shutil.copyfile(src=parfile, dst=f"./parfiles/{filename}.yml")
    except shutil.SameFileError:
        pass

    # Check parameters and modify e.g. metadata
    parameters = check_parameters(parameters=parameters)

    # Load model
    if parameters.get("preload_model"):
        try:
            model = sbm.PhaseNet.from_pretrained(parameters["preload_model"])
        except (ValueError, requests.exceptions.ConnectionError) as e:
            if os.path.isfile(parameters["preload_model"]) is True:
                model = torch.load(parameters["preload_model"], map_location=torch.device("cpu"))
            else:
                msg = f"{e}\nDid not find {parameters['preload_model']}."
                raise IOError(msg)
    else:
        phases = parameters.get("phases")
        if not phases:
            phases = "PSN"
        model = sbm.PhaseNet(phases=phases, norm="peak")
    # model = torch.compile(model)  # XXX Attribute error when saving model

    train_loader, val_loader, test = get_data_loaders(comm=subgroup_comm,
                                                      parameters=parameters,
                                                      model=model)

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        device = subgroup_comm.rank % GPUS_PER_NODE
    else:
        device = "cpu"

    # Auf Juwels muss als device "cuda" benutzt werden
    model = model.to("cuda")
    # Au Haicore werden alle gesehen, deswegen wird evice gewaehlt
    # model = model.to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)  # Wrap model with DDP.

    # Start training
    # specify loss function
    loss_fn = VectorCrossEntropyLoss()

    # specify learning rate and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Decaying learning rate
    if isinstance(parameters["learning_rate"], dict) and parameters["learning_rate"].get("decay") is True:
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=parameters["learning_rate"]["step_size"],
                                              gamma=parameters["learning_rate"]["gamma"])
    else:
        scheduler = None

    model, train_loss, val_loss = train_model(model=model,
                                              patience=parameters["patience"],
                                              epochs=parameters["epochs"],
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              train_loader=train_loader,
                                              validation_loader=val_loader,
                                              lr_scheduler=scheduler)

    # Return best validation loss as an individual's loss (trained so lower is better).
    dist.destroy_process_group()

    min_loss = min(val_loss)
    if is_nan(min_loss):
        return 1000
    return min_loss


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, path_checkpoint=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path_checkpoint (str, None): Path for the checkpoint to be saved to. If not None chechpoints are saved.
                            Default: None
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path_checkpoint
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')

        if self.path:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, validation_loader, loss_fn,
                optimizer=None, epochs=50, patience=5, lr_scheduler=None):
    """

    """
    # Initialize lists to track losses
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    # Load optimizer
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize early stopping class
    early_stopping = EarlyStopping(patience=patience, verbose=False, path_checkpoint=None)

    # Loop over each epoch to start training
    rank = dist.get_rank()
    for epoch in range(epochs):
        print(f"Start epoch {epoch + 1} for rank {rank}")
        # Train model (loop over each batch; batch_size is defined in DataLoader)
        # TODO (idea): test model with validation (compute metrics)
        train_loader.sampler.set_epoch(epoch)
        validation_loader.sampler.set_epoch(epoch)
        num_batches = len(train_loader)
        if rank == 0:
            # progress_bar = tqdm(total=num_batches + len(validation_loader),
            #                     desc=f"Epoch {epoch + 1}",
            #                     ncols=100,
            #                     bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]")
            progress_bar = contextlib.nullcontext()
        else:
            progress_bar = contextlib.nullcontext()

        with progress_bar as pbar:
            for batch_id, batch in enumerate(train_loader):
                # Compute prediction and loss
                pred = model(batch["X"].to(model.device))
                loss = loss_fn(y_pred=pred, y_true=batch["y"].to(model.device))

                # Do backpropagation
                optimizer.zero_grad()   # clear the gradients of all optimized variables
                loss.backward()         # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()        # perform a single optimization step (parameter update)

                # Compute loss for each batch and write loss to predefined lists
                dist.all_reduce(loss)
                loss /= dist.get_world_size()
                train_loss.append(loss.item())

                # # Update progressbar
                # if rank == 0:
                #     pbar.set_postfix({"loss": str(np.round(loss.item(), 4))})
                #     pbar.update()

            # Validate the model
            print(f"Validate epoch {epoch + 1} for rank {rank}")
            model.eval()     # Close the model for validation / evaluation
            with torch.no_grad():   # Disable gradient calculation
                for batch in validation_loader:
                    pred = model(batch["X"].to(model.device))
                    val_loss = loss_fn(pred, batch["y"].to(model.device))
                    dist.all_reduce(val_loss)
                    val_loss /= dist.get_world_size()
                    valid_loss.append(val_loss.item())

                    # if rank == 0:
                    #     pbar.set_postfix({"val_loss": str(np.round(val_loss.item(), 4))})
                    #     pbar.update()

            # Determine average training and validation loss
            avg_train_loss.append(sum(train_loss) / len(train_loss))
            avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

            # # Update progressbar
            # if rank == 0:
            #     pbar.set_postfix(
            #         {"loss": str(np.round(avg_train_loss[-1], 4)),
            #          "val_loss": str(np.round(avg_valid_loss[-1], 4))}
            #     )

        # Re-open model for next epoch
        model.train()

        # Clear training and validation loss lists for next epoch
        train_loss = []
        valid_loss = []

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss[-1], model)

        if early_stopping.early_stop:
            print("Validation loss does not decrease further. Early stopping")
            break

    return model, avg_train_loss, avg_valid_loss
