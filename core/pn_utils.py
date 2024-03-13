"""
Collection of functions to train and test PhaseNet
"""
import os
import torch
import pathlib
import seisbench

import numpy as np
import pandas as pd

from typing import Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from core.torch_functions import Metrics


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
    pred_pick_index = np.argmax(prediction[index, phase_index, true_pick - int(win_len_factor * sigma):   # TODO: Grenzwert von +- x * sigma
                                           true_pick + int(win_len_factor * sigma)]
                                .detach().numpy())
    # Add offset to pred_pick_index
    pred_pick_index += true_pick - int(win_len_factor * sigma)

    if pred_pick_index == 0:
        return None
    else:
        return pred_pick_index


def get_pick_probabilities(prediction_sample: (int, None), batch: dict, index: int,
                           phasenet_model: seisbench.models.phasenet.PhaseNet, phase: str = "P") -> Union[float, None]:

    phase_index = get_sb_phase_value(phase=phase, phasennet_model=phasenet_model)
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
                p_prob = get_pick_probabilities(batch=batch, prediction_sample=pred_p_samp,
                                                index=index, phase="P", phasenet_model=model)
                s_prob = get_pick_probabilities(batch=batch, prediction_sample=pred_s_samp,
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
               plot_residual_histogram: bool = False):
    """

    """
    test_generator = sbg.GenericGenerator(test_dataset)

    augmentations_test = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()), samples_before=int(parameters["nsamples"] / 3),
                               windowlen=parameters["nsamples"],
                               selection="first", strategy="variable"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=get_phase_dict(), sigma=parameters["sigma"], dim=0,
                                  model_labels=model.labels, noise_column=True)
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
