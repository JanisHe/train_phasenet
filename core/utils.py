import os
import shutil
import subprocess
import pathlib

import numpy as np
import pandas as pd
import obspy

import seisbench.data as sbd


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

    # Modifying metadata of datasets by adding fake events
    if parameters.get("add_fake_events"):
        for lst in parameters['datasets']:
            for dataset in lst.values():
                # Copy metadata file
                shutil.copyfile(src=os.path.join(dataset, "metadata.csv"),
                                dst=os.path.join(dataset, "tmp_metadata.csv"))
                metadata = pd.read_csv(os.path.join(dataset, "metadata.csv"))
                metadata_dct = metadata.to_dict(orient="list")
                num_add_events = int(len(metadata) * parameters["add_fake_events"] / 100)
                for i in range(num_add_events):
                    rand_data_index = np.random.randint(0, len(metadata))
                    for key in metadata_dct:
                        metadata_dct[key].append(metadata_dct[key][rand_data_index])
                # Convert back to dataframe
                metadata = pd.DataFrame(metadata_dct)
                metadata.to_csv(path_or_buf=os.path.join(dataset, "metadata.csv"))

    return parameters


def read_datasets(parameters: dict, dataset_key: str = "datasets"):
    """
    Read seisbench dataset from parameter file.
    """
    for lst in parameters[dataset_key]:
        for dataset_count, dataset in enumerate(lst.values()):
            if dataset_count == 0:
                sb_dataset = sbd.WaveformDataset(path=pathlib.Path(dataset),
                                                 sampling_rate=parameters["sampling_rate"])
            else:
                sb_dataset += sbd.WaveformDataset(path=pathlib.Path(dataset),
                                                  sampling_rate=parameters["sampling_rate"])

    return sb_dataset
