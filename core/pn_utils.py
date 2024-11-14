"""
Collection of functions to train and test PhaseNet
"""
import os
import torch
import pathlib
import seisbench  # noqa

import numpy as np
import pandas as pd

from typing import Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seisbench.generate as sbg # noqa
from seisbench.util import worker_seeding # noqa
from obspy.signal.trigger import trigger_onset

from core.torch_functions import Metrics


def get_phase_dict(num_phases=100):
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

    for i in range(num_phases):
        for phase in ["P", "S"]:
            map_phases.update({f"trace_{phase}_{i}_arrival_sample": phase})

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
                       phasennet_model: seisbench.models.phasenet.PhaseNet) -> int:
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

    true_pick_index = np.argmax(batch["y"][index, phase_index, start:end].detach().numpy())

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

                # Find true P and S phase arrival
                true_p_samp = get_true_pick(batch=batch,
                                            index=index,
                                            phase="P",
                                            phasenet_model=model,
                                            samples_before=samples_before,
                                            s_p=s_p)
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


def add_metrics(axes,
                metrics: Metrics):
    # TODO: Add mean and standard deviation
    textstr = (f"Precision: {np.round(metrics.precision, 2)}\n"
               f"Recall: {np.round(metrics.recall, 2)}\n"
               f"F1 score: {np.round(metrics.f1_score, 2)}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axes.text(x=0.05,
              y=0.95,
              s=textstr,
              transform=axes.transAxes,
              fontsize=10,
              verticalalignment='top',
              bbox=props)


def test_model(model: seisbench.models.phasenet.PhaseNet,
               test_dataset: seisbench.data.base.MultiWaveformDataset,
               plot_residual_histogram: bool = False,
               **parameters):
    """

    """
    test_generator = sbg.GenericGenerator(test_dataset)

    samples_before = int(parameters["nsamples"] / 3)
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()),
                               samples_before=samples_before,
                               windowlen=parameters["nsamples"],
                               selection="first",    # XXX Problem with multi events
                               strategy="move"),
        sbg.Normalize(demean_axis=-1,
                      amp_norm_axis=-1,
                      amp_norm_type=model.norm),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=get_phase_dict(),
                                  sigma=parameters["sigma"],
                                  dim=0,
                                  model_labels=model.labels,
                                  noise_column=True)
    ]
    test_generator.add_augmentations(augmentations)
    test_loader = DataLoader(dataset=test_generator,
                             batch_size=128,
                             shuffle=False,
                             num_workers=parameters["nworkers"],
                             worker_init_fn=worker_seeding,
                             drop_last=False)

    picks_and_probs = get_picks(model=model,
                                dataloader=test_loader,
                                sigma=parameters["sigma"],
                                win_len_factor=parameters["win_len_factor"],
                                threshold=parameters["true_pick_prob"],
                                samples_before=samples_before)

    # Evaluate metrics for P and S
    # 1. Determine true positives (tp), false positives (fp), and false negatives (fn) for P and S phases
    metrics_p = Metrics(predictions=picks_and_probs["pred_P"],
                        true_pick_prob=parameters["true_pick_prob"],
                        arrival_residual=parameters["arrival_residual"])

    metrics_s = Metrics(predictions=picks_and_probs["pred_S"],
                        true_pick_prob=parameters["true_pick_prob"],
                        arrival_residual=parameters["arrival_residual"])

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