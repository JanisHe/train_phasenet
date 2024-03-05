"""
Collection of functions to create seisbench datasets
"""
import os.path

import pandas as pd
import random
import obspy
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from utils import snr_pick
import lxml
import seisbench
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import is_nan
from typing import Union
import pathlib


class Metrics:
    """

    """
    def __init__(self, probabilities, residuals,
                 true_pick_prob=0.5, arrival_residuals=10):
        self.probabilities = probabilities
        self.residuals = residuals
        self.true_pick_prob = true_pick_prob
        self.arrival_residuals = arrival_residuals

        self.predictions = None
        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

    def true_false_positives(self, predictions) -> (float, float, float):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        for index, prediction in enumerate(predictions):
            if not is_nan(prediction):
                if (self.probabilities[index] >= self.true_pick_prob and
                        abs(self.residuals[index]) <= self.arrival_residuals):
                    self.true_positive += 1
                elif (self.probabilities[index] >= self.true_pick_prob and
                      abs(self.residuals[index]) > self.arrival_residuals):
                    self.false_positive += 1
                elif (self.probabilities[index] < self.true_pick_prob and
                      abs(self.residuals[index]) > self.arrival_residuals):
                    self.false_negative += 1

    def precision(self, predictions=None) -> float:
        self.check_predictions(predictions=predictions)
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self, predictions=None) -> float:
        self.check_predictions(predictions=predictions)
        return self.true_positive / (self.true_positive + self.false_negative)

    def f1_score(self, predictions=None) -> float:

        return 2 * ((self.precision(predictions=predictions) * self.recall(predictions=predictions)) / (
                    self.precision(predictions=predictions) + self.recall(predictions=predictions)))

    def check_predictions(self, predictions):
        if not self.true_positive:
            self.predictions = predictions
            self.true_false_positives(predictions=predictions)


def get_event_params(event):
    """
    Get parameter from event for SeisBench datasets.
    If parameter can not be found it is set to None

    :param event:
    :return:
    """
    origin = event.origins[0]
    try:
        magnitudes = event.magnitudes[0]
    except IndexError:
        magnitudes = None
    source_id = str(event.resource_id)

    event_params = {
        "source_id": source_id,
        "source_origin_time": str(origin.time),
        "source_origin_uncertainty_sec": origin.time_errors.get("uncertainty"),
        "source_latitude_deg": origin.latitude,
        "source_latitude_uncertainty_km": origin.latitude_errors.get("uncertainty"),
        "source_longitude_deg": origin.longitude,
        "source_longitude_uncertainty_km": origin.longitude_errors.get("uncertainty"),
        "source_depth_km": origin.depth / 1e3,
        "source_depth_uncertainty_km": origin.depth_errors.get("uncertainty"),
    }

    if event_params["source_depth_uncertainty_km"]:
        event_params["source_depth_uncertainty_km"] = event_params["source_depth_uncertainty_km"] / 1e3

    if magnitudes is not None:
        event_params["source_magnitude"] = magnitudes.mag
        event_params["source_magnitude_uncertainty"] = magnitudes.mag_errors.get("uncertainty")
        event_params["source_magnitude_type"] = magnitudes.magnitude_type
        event_params["source_magnitude_author"] = magnitudes.resource_id

    return event_params


def get_trace_params(network_code: str,
                     station_code: str,
                     starttime: obspy.UTCDateTime,
                     endtime: obspy.UTCDateTime,
                     fdsn_client=None,
                     inventory=None):
    """

    :param network_code:
    :param station_code:
    :param starttime:
    :param endtime:
    :param fdsn_client:
    :param inventory:
    :return:
    """
    if fdsn_client and inventory is None:
        try:
            inventory = fdsn_client.get_stations(
                network=network_code,
                station=station_code,
                starttime=starttime,
                endtime=endtime,
                level="response"
            )
        except (FDSNException, lxml.etree.XMLSyntaxError):
            pass

    trace_params = dict(
        station_network_code=network_code,
        station_code=station_code,
        trace_channel=None,
        station_location_code=None
    )

    if inventory:
        trace_params.update(
            {"station_longitude_deg": inventory[0].stations[0].longitude,
             "station_latitude_deg": inventory[0].stations[0].latitude}
        )

    return trace_params


def get_waveforms(picks,
                  trace_params: dict,
                  clients: list,
                  time_before=60.0,
                  time_after=60.0,
                  channel="*",
                  min_samples=3001):
    """

    :param picks:
    :param trace_params:
    :param clients:
    :param time_before:
    :param time_after:
    :param channel:
    :param min_samples:
    :return:
    """
    starttime = list(picks.values())[0] - time_before  # Starttime from earliest pick
    endtime = list(picks.values())[-1] + time_after     # Endtime from latest pick

    # Get channels from trace_params (overrides channels from arguments)
    if trace_params.get("trace_channel"):
        channel = f"{trace_params.get('trace_channel')}*"

    # Try to get waveforms from all clients in list
    for client in clients:
        try:
            stream = client.get_waveforms(
                network=trace_params["station_network_code"],
                station=trace_params["station_code"],
                location="*",
                channel=channel,
                starttime=starttime,
                endtime=endtime
            )

            if len(stream) > 0:
                break
        except BaseException:
            stream = obspy.Stream()

    if len(stream) == 0:
        msg = "No traces in waveforms."
        raise ValueError(msg)
    else:
        # Update trace_params
        trace_params['trace_channel'] = stream[0].stats.channel[:2]
        # TODO: Location is correct in editor afterwards
        trace_params["station_location_code"] = stream[0].stats.location

        # Check that the traces have the same sampling rate
        sampling_rate = stream[0].stats.sampling_rate
        try:
            assert all(trace.stats.sampling_rate == sampling_rate for trace in stream)
        except AssertionError:
            stream.resample(sampling_rate=sampling_rate)

        # Check min length of all waveforms
        min_num_samples = (time_before + time_after) * sampling_rate
        if min_num_samples >= min_samples:
            try:
                assert all(trace.stats.npts >= min_num_samples for trace in stream)
            except AssertionError:
                msg = f"One trace in {stream} does not have the correct length."
                raise ValueError(msg)
        else:
            msg = "Traces in {stream} do not fullfill the minimal length criterium"
            raise ValueError(msg)

        # Update sampling_rate in trace_params and network_code
        trace_params.update({"trace_sampling_rate_hz": sampling_rate})
        trace_params["station_network_code"] = stream[0].stats.network

        return stream, trace_params


def cross_validation(metdata_pathname,
                     train=0.7,
                     dev=0.2,
                     test=0.1):
    """

    :param metdata_pathname:
    :param train:
    :param dev:
    :param test:
    :return:
    """
    # TODO: if split is already used as column, only change train and dev for cross-validation
    if abs(1 - sum([train, dev, test])) >= 1e-3:
        msg = "Sum of train, dev, test must be 1!"
        raise ValueError(msg)

    # Read metadata csv file
    metadata = pd.read_csv(metdata_pathname)

    # Define arbitrary list with values of train/dev/test with length of metadata number of rows
    # TODO: Split test to same event(s) and not randomly
    test_lst = ["test"] * int(metadata.shape[0] * test)
    dev_lst = ["dev"] * int(metadata.shape[0] * dev)
    train_lst = ["train"] * (metadata.shape[0] - sum([len(test_lst), len(dev_lst)]))
    train_dev_test_lst = test_lst + dev_lst + train_lst

    # Shuffle train_dev_test_lst
    random.shuffle(train_dev_test_lst)

    # Add column with split values to metadata
    metadata["split"] = train_dev_test_lst

    # Save new metadata
    metadata.to_csv(path_or_buf=metdata_pathname)


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


def get_sb_phase_value(phase: str):
    if phase == "P":
        return 0
    elif phase == "S":
        return 1
    else:
        raise ValueError(f"Phase {phase} is not allowed.")


def get_true_pick(batch: dict, index: int, phase: str = "P") -> Union[str, None]:
    phase_index = get_sb_phase_value(phase=phase)
    true_pick = np.where(batch["y"][index, phase_index, :] == np.max(np.array(batch["y"][index, phase_index, :])))[0]

    if len(true_pick) == 1:
        return true_pick[0]
    else:
        return None


def get_predicted_pick(prediction: torch.Tensor, index: int, true_pick: (int, None),
                       sigma=30, phase: str = "P", win_len_factor=10) -> Union[int, None]:
    # Return None if true_pick is None
    if not true_pick:
        return None

    # Get phase index
    phase_index = get_sb_phase_value(phase=phase)

    # Find maximum of predicted pick
    pred_pick = np.where(prediction[index, phase_index, :] ==
                         np.max(np.array(prediction[index, phase_index,
                                         true_pick - int(win_len_factor * sigma):   # TODO: Grenzwert von +- x * sigma
                                         true_pick + int(win_len_factor * sigma)]
                                         )))[0]

    if len(pred_pick) == 1:
        return pred_pick[0]
    else:
        return None


def get_pick_probabilities(prediction_sample: (int, None), batch: dict, index: int,
                           phase: str = "P") -> Union[float, None]:

    phase_index = get_sb_phase_value(phase=phase)
    if prediction_sample:
        return batch["y"][index, phase_index, prediction_sample].item()
    else:
        return None


def pick_residual(true: int, predicted: int) -> (int, np.nan):
    if not true or not predicted:
        return np.nan
    else:
        return predicted - true


def get_picks(model, dataloader, sigma=30, win_len_factor=10):
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
                true_p_samp = get_true_pick(batch=batch, index=index, phase="P")
                true_s_samp = get_true_pick(batch=batch, index=index, phase="S")

                # Find predicted P and S arrival
                pred_p_samp = get_predicted_pick(prediction=pred, index=index, true_pick=true_p_samp,
                                                 sigma=sigma, phase="P", win_len_factor=win_len_factor)
                pred_s_samp = get_predicted_pick(prediction=pred, index=index, true_pick=true_s_samp,
                                                 sigma=sigma, phase="S", win_len_factor=win_len_factor)

                # Get pick probabilities for P and S
                p_prob = get_pick_probabilities(batch=batch, prediction_sample=pred_p_samp,
                                                index=index, phase="P")
                s_prob = get_pick_probabilities(batch=batch, prediction_sample=pred_s_samp,
                                                index=index, phase="S")

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

    return pick_results


def residual_histogram(residuals, axes, bins=60, xlim=(-100, 100)):
    counts, bins = np.histogram(residuals, bins=bins, range=xlim)
    axes.hist(bins[:-1], bins, weights=counts, edgecolor="b")

    return axes


def add_metrics(axes, metrics: Metrics):
    textstr = (f"Precision: {np.round(metrics.precision(), 2)}\n"
               f"Recall: {np.round(metrics.recall(), 2)}\n"
               f"F1 score: {np.round(metrics.f1_score(), 2)}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axes.text(0.05, 0.95, textstr, transform=axes.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)


def test_model(model: seisbench.models.phasenet.PhaseNet,
               test_dataset: seisbench.data.base.MultiWaveformDataset,
               parameters: dict,
               plot_residual_histogram: bool=False):
    """

    """
    test_generator = sbg.GenericGenerator(test_dataset)

    augmentations_test = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()), samples_before=int(parameters["nsamples"] / 3),
                               windowlen=parameters["nsamples"],
                               selection="first", strategy="variable"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=get_phase_dict(), sigma=parameters["sigma"], dim=0)
    ]
    test_generator.add_augmentations(augmentations_test)
    test_loader = DataLoader(dataset=test_generator, batch_size=parameters["batch_size"],
                             shuffle=False, num_workers=parameters["nworkers"],
                             worker_init_fn=worker_seeding, drop_last=False)

    picks_and_probs = get_picks(model=model, dataloader=test_loader, sigma=parameters["sigma"],
                                win_len_factor=parameters["win_len_factor"])

    # Evaluate metrics for P and S
    # 1. Determine true positives (tp), false positives (fp), and false negatives (fn) for P and S phases
    metrics_p = Metrics(probabilities=picks_and_probs["prob_P"], residuals=picks_and_probs["residual_P"],
                        true_pick_prob=parameters["true_pick_prob"], arrival_residuals=parameters["arrival_residual"])
    metrics_s = Metrics(probabilities=picks_and_probs["prob_S"], residuals=picks_and_probs["residual_S"],
                        true_pick_prob=parameters["true_pick_prob"], arrival_residuals=parameters["arrival_residual"])

    precision_p = metrics_p.precision(predictions=picks_and_probs["pred_P"])
    precision_s = metrics_s.precision(predictions=picks_and_probs["pred_S"])
    recall_p = metrics_p.recall()
    recall_s = metrics_s.recall()
    f1_p = metrics_p.f1_score()
    f1_s = metrics_s.f1_score()

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

    return precision_p, precision_s, recall_p, recall_s, f1_p, f1_s
