import os
import shutil
import subprocess
import pathlib
import warnings
import torch
import logging
import time

import numpy as np
import pandas as pd
import obspy
import socket

import datetime as dt
import seisbench.data as sbd
from tqdm import tqdm
from mpi4py import MPI
import torch.distributed as dist


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


def is_nan(num):
    return num != num


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
    master_address = os.environ["MASTER_ADDR"]
    # Each rank needs to get the hostname of rank 0 of its group.
    # master_address = comm.bcast(str(master_address), root=0)

    # Save environment variables.
    # os.environ["MASTER_ADDR"] = master_address
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

