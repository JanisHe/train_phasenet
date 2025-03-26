import os
import subprocess
import pathlib

import numpy as np
import pandas as pd
import torch
import obspy
import seisbench  # noqa

from typing import Union
import seisbench.data as sbd # noqa
import seisbench.generate as sbg # noqa
from seisbench.util import worker_seeding # noqa
from seisbench.models.phasenet import PhaseNet # noqa
from obspy.signal.trigger import trigger_onset


def rms(x: np.array) -> float:
    """
    Root-mean-square of array x
    :param x: numpy array with a dimension of 1
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal: np.array,
                          noise: np.array,
                          decibel: bool=True) -> float:
    """
    Returns signal-to-noise ratio (SNR) from given signal array and noise array.

    :param signal: np.array for the signal
    :param noise: np.array for noise
    :param decibel: If True, SNR is returned in decibel (dB), default is True
    """
    if decibel is True:
        return 20 * np.log10(rms(signal) / rms(noise))
    else:
        return rms(signal) / rms(noise)


def snr(signal: np.array,
        noise: np.array,
        decibel: bool=True) -> float:
    """
    Wrapper for signal-to-noise_ratio
    """
    return signal_to_noise_ratio(signal=signal,
                                 noise=noise,
                                 decibel=decibel)


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
    """
    Proves whether a given element num is nan (not a number)
    Returns True if num is nan, otherwise False
    """
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

    # Check for drop_rate, stride, depth, kernel_size, and filter_root in parameters
    # These parameters are used to modify the PhaseNet network
    if not parameters.get("sampling_rate"):
        parameters["sampling_rate"] = 100

    if not parameters.get("in_channels"):
        parameters["in_channels"] = 3

    if not parameters.get("drop_rate"):
        parameters["drop_rate"] = 0

    if not parameters.get("stride"):
        parameters["stride"] = 4

    if not parameters.get("kernel_size"):
        parameters["kernel_size"] = 7

    if not parameters.get("filters_root"):
        parameters["filters_root"] = 8

    if not parameters.get("depth"):
        parameters["depth"] = 5

    if not parameters.get("filter_factor"):
        parameters["filter_factor"] = 1

    if not parameters.get("learning_rate"):
        parameters["learning_rate"] = 0.01

    if not parameters.get("batch_size"):
        parameters["batch_size"] = 256

    if not parameters.get("nsamples"):
        parameters["nsamples"] = 3001

    if not parameters.get("phases"):
        parameters["phases"] = "PSN"

    # Update paramaters to train single phase models
    parameters["p_phase"] = False
    parameters["s_phase"] = False
    if "P" in parameters["phases"]:
        parameters["p_phase"] = True
    if "S" in parameters["phases"]:
        parameters["s_phase"] = True

    if not parameters.get("activation_function"):
        parameters["activation_function"] = "relu"

    return parameters


def filter_dataset(filter: dict,
                   dataset: sbd.WaveformDataset):
    """
    Filtering metadata of seisbench dataset. Since filtering is inplace, nothing is returned.
    keywords of filter:
    - operation: "<" or ">"
    - item: which column in metadata
    - threshold: value for threshold
    """
    try:
        if filter["operation"] == "<":
            mask = dataset.metadata[filter["item"]] < filter["threshold"]
        elif filter["operation"] == ">":
            mask = dataset.metadata[filter["item"]] > filter["threshold"]
        else:
            msg = f'Filter operation {filter["operation"]} is not known'
            raise ValueError(msg)

        dataset.filter(mask, inplace=True)
    except KeyError:
        print(f"No filtering for data set {dataset} possible.")


def read_datasets(parameters: dict,
                  component_order: str = "ZNE",
                  dataset_key: str = "datasets",
                  filter: (dict, None) = None,
                  verbose=True,
                  trace_func=print):
    """
    Read seisbench dataset from parameter file.
    """
    dataset_count = 0
    for dataset in parameters[dataset_key]:  # Loop over each entry in list
        for dataset_name in dataset.keys():  # Loop over each key in dataset (depends on type (dict | list)
            if dataset_count == 0:
                sb_dataset = sbd.WaveformDataset(path=pathlib.Path(dataset[dataset_name]),
                                                 sampling_rate=parameters["sampling_rate"],
                                                 component_order=component_order)
                if filter:
                    filter_dataset(filter=filter, dataset=sb_dataset)
            else:
                subset = sbd.WaveformDataset(path=pathlib.Path(dataset[dataset_name]),
                                             sampling_rate=parameters["sampling_rate"],
                                             component_order=component_order)
                if filter:
                    filter_dataset(filter=filter, dataset=subset)
                sb_dataset += subset

            dataset_count += 1
            if verbose is True:
                trace_func(f"Successfully read dataset {dataset_name}")

    return sb_dataset


def add_fake_events_to_metadata(metadata: pd.DataFrame,
                                percentage: int):
    # Convert metadata dataframe to dictionary
    metadata_dct = metadata.to_dict(orient="list")

    # Determine number of fake events from percentage and shape of metadata
    number = int(len(metadata) * percentage / 100)

    # Randomly add events to metadata by copying rows from metadata and append to dictionary
    for i in range(number):
        rand_data_index = np.random.randint(low=0,
                                            high=len(metadata))
        for key in metadata_dct:
            metadata_dct[key].append(metadata_dct[key][rand_data_index])

    # Convert dictionary back to pandas dataframe and return
    return pd.DataFrame(metadata_dct)


def add_fake_events(sb_dataset: sbd.WaveformDataset,
                    percentage: int):
    """
    Copying metadata to add fake events
    """
    # Convert metadata to dictionary
    if hasattr(sb_dataset, "datasets") is False:
        sb_dataset._metadata = add_fake_events_to_metadata(metadata=sb_dataset.metadata,
                                                           percentage=percentage)
    elif hasattr(sb_dataset, "datasets") is True:
        metadata_collection = []
        for dataset in sb_dataset.datasets:  # Loop over each dataset
            metadata = add_fake_events_to_metadata(metadata=dataset.metadata,
                                                   percentage=percentage)
            metadata_collection.append(metadata)
            dataset._metadata = metadata

        # Update metadata of full dataset
        sb_dataset._metadata = pd.concat(objs=metadata_collection)


def get_phase_dict(num_phases=100,
                   p_phase=True,
                   s_phase=True):
    """
    Returns dictionary with different phase names that are mapped only to P and S.
    If p_phase and s_phase are True, then both phase types are inlcuded, otherwise the phase which is set to False
    is ignored.
    """
    # Allocate empty dicts for P and S
    map_phases_p = {}
    map_phases_s = {}

    if p_phase:
        map_phases_p = {
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
        }

        # Update for multiple P arrivals within a single window
        for i in range(num_phases):
            map_phases_p.update({f"trace_P_{i}_arrival_sample": "P"})

    if s_phase:
        map_phases_s = {
            "trace_s_arrival_sample": "S",
            "trace_S_arrival_sample": "S",
            "trace_S1_arrival_sample": "S",
            "trace_Sg_arrival_sample": "S",
            "trace_SmS_arrival_sample": "S",
            "trace_Sn_arrival_sample": "S",
            "trace_ASg_arrival_sample": "S",
        }

        # Update for multiple S arrivals within a single window
        for i in range(num_phases):
            map_phases_s.update({f"trace_S_{i}_arrival_sample": "S"})

    # Concatenate boh dictionaries for P and S and return final map_phases dict
    map_phases = {**map_phases_p, **map_phases_s}

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


def get_sb_phase_value(phase: str,
                       phasennet_model: PhaseNet) -> int:
    """
    Reads phases labels from seisbench model and returns the index of the phase
    """
    try:
        return phasennet_model.labels.index(phase)
    except ValueError:
        ValueError(f"Phase {phase} is not in model labels ({phasennet_model.labels}).")


def get_true_pick(batch: dict,
                  index: int,
                  phasenet_model: seisbench.models.phasenet.PhaseNet,
                  phase: str = "P",
                  samples_before: int=1000,
                  s_p: int=200) -> Union[str, None]:
    phase_index = get_sb_phase_value(phase=phase,
                                     phasennet_model=phasenet_model)

    # Define search space for true P- and S-arrival
    start = int(samples_before - s_p)
    end = int(start + 3 * s_p)

    # Set limits for start and end
    if start < 0:
        start = 0
    if end > len(batch["y"][index, phase_index, :]):
        end = len(batch["y"][index, phase_index, :])

    # Try except statement is required for single phase models
    try:
        true_pick_index = np.argmax(batch["y"][index, phase_index, start:end].detach().numpy())
    except ValueError:
        true_pick_index = 0

    true_pick_index = int(true_pick_index + start)

    if true_pick_index <= 0:
        return None
    else:
        return true_pick_index


def get_predicted_pick(prediction: torch.Tensor,
                       index: int,
                       true_pick: (int, None),
                       phasenet_model: seisbench.models.phasenet.PhaseNet,
                       sigma=30,
                       phase: str = "P",
                       win_len_factor=10,
                       threshold: float=0.3) -> list:

    # Return None if true_pick is None
    if not true_pick:
        return None

    # Get phase index
    phase_index = get_sb_phase_value(phase=phase,
                                     phasennet_model=phasenet_model)

    # If GPU is used, convert prediction from cuda to numpy
    prediction = prediction.cpu()

    # Find maximum of predicted pick
    lower_bound = true_pick - int(win_len_factor * sigma)
    upper_bound = true_pick + int(win_len_factor * sigma)
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > prediction.shape[2]:
        upper_bound = prediction.shape[2]
    prediction_window = prediction[index, phase_index, lower_bound:upper_bound].detach().numpy()

    # Get picks in window
    # Methods is copied from seisbench.models.base.WaveformModel.picks_from_annotations
    picks = []
    triggers = trigger_onset(charfct=prediction[index, phase_index, lower_bound:upper_bound].detach().numpy(),
                             thres1=threshold,
                             thres2=threshold / 2)
    for s0, s1 in triggers:
        peak_value = np.max(prediction_window[s0 : s1 + 1])
        s_peak = s0 + np.argmax(prediction_window[s0 : s1 + 1])
        picks.append({"phase": phase,
                      "peak_value": peak_value,
                      "sample": s_peak + lower_bound,
                      "residual": abs(s_peak + lower_bound - true_pick)})


    return picks


def get_pick_probabilities(prediction_sample: (int, None),
                           prediction: torch.Tensor,
                           index: int,
                           phasenet_model: seisbench.models.phasenet.PhaseNet,
                           phase: str = "P") -> Union[float, None]:

    phase_index = get_sb_phase_value(phase=phase, phasennet_model=phasenet_model)
    if prediction_sample:
        return prediction[index, phase_index, prediction_sample].item()
    else:
        return None


def pick_residual(true: int,
                  predicted: int) -> (int, np.nan):
    if not true or not predicted:
        return np.nan
    else:
        return predicted - true


def get_picks(model,
              dataloader,
              sigma: int=30,
              win_len_factor: int=10,
              threshold: float=0.1,
              samples_before: int=0):

    pick_results = {"true_P": np.empty(len(dataloader.dataset)),
                    "true_S": np.empty(len(dataloader.dataset)),
                    "pred_P": [],
                    "pred_S": [],
                    }

    # Map main phase arrivals, i.e. trace_P_arrival_sample and trace_S_arrival_sample, to P and S
    metadata = map_arrivals(dataframe=dataloader.dataset.dataset.metadata)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            pred = model(batch["X"].to(model.device))
            for index in range(pred.shape[0]):
                # Determine S-P time
                try:
                    s_p = metadata.loc[index + (batch_index * dataloader.batch_size), "S"] - \
                          metadata.loc[index + (batch_index * dataloader.batch_size), "P"]
                except KeyError:
                    s_p = 250

                if is_nan(s_p):
                    s_p = 250

                # Find true P and S phase arrival
                true_p_samp = None  # Default values for true P and S arrival
                true_s_samp = None  # which is required e.g. for single phase models
                if "P" in model.labels:
                    true_p_samp = get_true_pick(batch=batch,
                                                index=index,
                                                phase="P",
                                                phasenet_model=model,
                                                samples_before=samples_before,
                                                s_p=s_p)

                if "S" in model.labels:
                    true_s_samp = get_true_pick(batch=batch,
                                                index=index,
                                                phase="S",
                                                phasenet_model=model,
                                                samples_before=samples_before,
                                                s_p=s_p)

                # Find predicted P and S arrival
                pred_p_samp = get_predicted_pick(prediction=pred,
                                                 index=index,
                                                 true_pick=true_p_samp,
                                                 sigma=sigma,
                                                 phase="P",
                                                 win_len_factor=win_len_factor,
                                                 phasenet_model=model,
                                                 threshold=threshold)

                pred_s_samp = get_predicted_pick(prediction=pred,
                                                 index=index,
                                                 true_pick=true_s_samp,
                                                 sigma=sigma,
                                                 phase="S",
                                                 win_len_factor=win_len_factor,
                                                 phasenet_model=model,
                                                 threshold=threshold)

                # Write results to dictionary
                pick_results["true_P"][index + (batch_index * dataloader.batch_size)] = true_p_samp
                pick_results["true_S"][index + (batch_index * dataloader.batch_size)] = true_s_samp
                pick_results["pred_P"].append(pred_p_samp)
                pick_results["pred_S"].append(pred_s_samp)

    return pick_results


def residual_histogram(residuals,
                       axes,
                       bins=60,
                       xlim=(-100, 100)):

    counts, bins = np.histogram(residuals,
                                bins=bins,
                                range=xlim)
    axes.hist(bins[:-1], bins,
              weights=counts,
              edgecolor="b")

    return axes


def check_propulate_limits(params: dict) -> dict:
    """
    Check whether one parameter in dictionary params has only a length of one.
    If yes, the same value is appended to the tuple. If only one parameter is in params,
    this parameters is not modified by propulate to find the best hyperparameters.

    This function is necessary to run propulate sucessfull.
    """
    for key, value in params.items():
        if isinstance(value, tuple):
            if len(value) == 1:
                params[key] = tuple([value[0], value[0]])
        elif isinstance(value, list):
            if len(value) == 1:
                params[key] = tuple([value[0], value[0]])
            else:
                params[key] = tuple(value)
        else:
            params[key] = tuple([value, value])

    return params


def best_threshold(recall_precision_array: np.ndarray,
                   thresholds: np.ndarray,
                   optimal_value=np.array([1, 1])):
    distances = np.zeros(recall_precision_array.shape[0])
    for index, element in enumerate(recall_precision_array):
        distances[index] = np.linalg.norm(element - optimal_value)

    # Return probability for minimum of distances
    return thresholds[np.argmin(distances)]


if __name__ == "__main__":
    params = {
        "learning_rate": [0.01],
        "batch_size": [8, 16, 32, 64],
        "activation_function": ["relu", "elu"],
    }

    limits = check_propulate_limits(params=params)
    print(limits)