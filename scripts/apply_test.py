import os
import copy
import pathlib

import obspy
import tqdm

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import seisbench  # noqa
import seisbench.models as sbm  # noqa
import seisbench.generate as sbg # noqa
import seisbench.data as sbd  # noqa
from seisbench.util import worker_seeding # noqa
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from typing import Union
from sklearn.metrics import auc
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient

from core.utils import is_nan, read_datasets, get_phase_dict, get_picks, best_threshold, event_picks, load_stations
from core.torch_functions import test_model


def main(parfile):
    """
    Test model from parfile
    """
    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    # Add filename for test to parameters
    # Load model
    if parameters.get("model"):
        parameters["filename"] = pathlib.Path(parameters["model"]).stem
        model = torch.load(parameters.pop("model"), map_location=torch.device("cpu"))
    elif parameters.get("preload_model"):
        model = sbm.PhaseNet.from_pretrained(parameters["preload_model"])
        parameters["filename"] = parameters["preload_model"]
    else:
        msg = "Whether model nor preload_model is definde in parfile."
        raise ValueError(msg)

    # Read datasets
    seisbench_dataset = read_datasets(parameters=parameters, dataset_key="datasets")

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Test model on test data from dataset
    metrics_p, metrics_s = test_model(model=model, test_dataset=test, plot_residual_histogram=True, **parameters)

    print("Precision P:", metrics_p.precision)
    print("Precision S:", metrics_s.precision)
    print("Recall P:", metrics_p.recall)
    print("Recall S:", metrics_s.recall)
    print("F1 P:", metrics_p.f1_score)
    print("F1 S:", metrics_s.f1_score)


def misclassified_data(parfile, probability=None):
    """
    Finding missclassifications from test data sets
    """
    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    # Load model
    if parameters.get("model"):
        parameters["filename"] = pathlib.Path(parameters["model"]).stem
        model = torch.load(parameters.pop("model"), map_location=torch.device("cpu"))
    elif parameters.get("preload_model"):
        model = sbm.PhaseNet.from_pretrained(parameters["preload_model"])
        parameters["filename"] = parameters["preload_model"]
    else:
        msg = "Whether model nor preload_model is defined in parfile."
        raise ValueError(msg)

    # Read datasets
    seisbench_dataset = read_datasets(parameters=parameters, dataset_key="datasets")

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Test model for given probability in parameters or given probability from arguments
    if probability:
        parameters["true_pick_prob"] = probability

    test_generator = sbg.GenericGenerator(test)

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

    picks_and_probs = get_picks(model=model,
                                dataloader=test_loader,
                                sigma=parameters["sigma"],
                                win_len_factor=parameters["win_len_factor"],
                                return_data=True)
    
    # Finding wrong classified data, i.e. finding false positives and negatives
    for index, (pred_sample, pred_prob, residual) in enumerate(zip(picks_and_probs["pred_P"], picks_and_probs["prob_P"], picks_and_probs["residual_P"])):
        if not is_nan(pred_sample):
            if pred_prob >= parameters["true_pick_prob"] and abs(residual) > parameters["arrival_residual"]:
                print(picks_and_probs["true_P"][index], residual, "FP")
                ax1 = plt.subplot(211)
                ax1.plot(picks_and_probs["data"][index, :, :].T)
                ax2 = plt.subplot(212, sharex=ax1)
                ax2.plot(picks_and_probs["prediction"][index, :, :].T)
                plt.show()
            elif pred_prob < parameters["true_pick_prob"] and abs(residual) > parameters["arrival_residual"]:
                print(picks_and_probs["true_P"][index], residual, "FN")
                ax1 = plt.subplot(211)
                ax1.plot(picks_and_probs["data"][index, :, :].T)
                ax2 = plt.subplot(212, sharex=ax1)
                ax2.plot(picks_and_probs["prediction"][index, :, :].T)
                plt.show()


def probabilities(parfile,
                  probs: Union[np.array, None] = None,
                  model_path: str = None):
    """
    Compute precision and recall for different probabilities

    :param model_path: If set models are loaded from model_path and parfile from training can be used
    """
    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    # Check device for model
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    if not model_path:
        if parameters.get("model"):
            if os.path.isfile(parameters["model"]) is True:
                parameters["filename"] = pathlib.Path(parameters["model"]).stem
                model = torch.load(parameters.pop("model"),
                                   map_location=torch.device(device))
            else:
                parameters["filename"] = parameters["model"]
                model = sbm.PhaseNet.from_pretrained(parameters.pop("model"))
        else:
            msg = "No model is defined."
            raise ValueError(msg)
    else:
        parameters["filename"] = pathlib.Path(parameters["model_name"]).stem
        model = torch.load(os.path.join(model_path, parameters.pop("model_name")),
                           map_location=torch.device(device))

    # Add sampling_rate from model to parameters
    parameters["sampling_rate"] = model.sampling_rate

    # Add phases to parameters
    parameters["p_phase"] = False
    parameters["s_phase"] = False
    if "P" in model.labels:
        parameters["p_phase"] = True
    if "S" in model.labels:
        parameters["s_phase"] = True

    # Read datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      dataset_key="datasets")

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Test model on test data from dataset
    # Loop over true_pick_probabilities
    if isinstance(probs, np.ndarray) is False:
        probs = np.linspace(1e-3, 0.99, num=10)

    precisions_p, precisions_s = [],  []
    recalls_p, recalls_s = [], []
    f1_p, f1_s = [], []
    tp_p, fp_p, fn_p = [], [], []
    tp_s, fp_s, fn_s = [], [], []
    with tqdm.tqdm(total=len(probs), desc=f"Testing model", ncols=100,
                   bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]") as pbar:
        for prob in probs:
            parameters["true_pick_prob"] = prob
            metrics_p, metrics_s = test_model(model=model,
                                              test_dataset=test,
                                              plot_residual_histogram=False,
                                              **parameters)

            if metrics_p:
                precisions_p.append(metrics_p.precision)
                recalls_p.append(metrics_p.recall)
                f1_p.append(metrics_p.f1_score)
                tp_p.append(metrics_p.true_positive)
                fp_p.append(metrics_p.false_positive)
                fn_p.append(metrics_p.false_negative)

            if metrics_s:
                precisions_s.append(metrics_s.precision)
                recalls_s.append(metrics_s.recall)
                f1_s.append(metrics_s.f1_score)
                tp_s.append(metrics_s.true_positive)
                fp_s.append(metrics_s.false_positive)
                fn_s.append(metrics_s.false_negative)

            pbar.update()

    # Finding the best thresholds for P and S from precision-recall curve
    # Determining distances of each precision-recall pair to point (1, 1)
    # Smallest distance represents the best threshold
    best_threshold_p = 999  # Default values, required for single phase models
    best_threshold_s = 999
    if metrics_p:
        rp_p = np.array(list(zip(recalls_p, precisions_p)))  # recall-precision array for P
        best_threshold_p = best_threshold(recall_precision_array=rp_p,
                                          thresholds=probs)

        f1_p_best = {"best_f1_p": np.max(f1_p),
                     "idx": np.argmax(f1_p)}
        try:  # Determining area under precision-recall curve for P
            auc_p = auc(x=recalls_p,
                        y=precisions_p)
        except ValueError:
            auc_p = 999
    if metrics_s:
        rp_s = np.array(list(zip(recalls_s, precisions_s)))  # recall-precision array for S
        best_threshold_s = best_threshold(recall_precision_array=rp_s,
                                          thresholds=probs)

        f1_s_best = {"best_f1_s": np.max(f1_s),
                     "idx": np.argmax(f1_s)}
        try:  # Determining area under precision-recall curve for S
            auc_s = auc(x=recalls_s,
                        y=precisions_s)
        except ValueError:
            auc_s = 999

    # Plot
    fig= plt.figure(figsize=(11, 5))
    ax_pr = fig.add_subplot(121)
    if metrics_p:
        ax_pr.plot(recalls_p, precisions_p, label=f"P (AUC: {auc_p:.2f})")

    if metrics_s:
        ax_pr.plot(recalls_s, precisions_s, label=f"S (AUC: {auc_s:.2f})")

    ax_pr.legend()
    ax_pr.grid(visible=True)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0.75, 1.05)
    # Box for best model parameters
    text_box = AnchoredText(s=f"Optimal P threshold: {best_threshold_p:.2f}\n"
                              f"Optimal S threshold: {best_threshold_s:.2f}",
                            frameon=False,
                            loc="lower left",
                            pad=0.5)
    plt.setp(text_box.patch,
             facecolor='white',
             alpha=0.5)
    ax_pr.add_artist(text_box)

    # Set probabilities in PR curve
    # for index, prob in enumerate(probs):
    #     ax_pr.text(x=recalls_p[index],
    #                y=precisions_p[index],
    #                s=str(np.round(prob, 2)))

    ax = fig.add_subplot(122)
    if metrics_p:
        ax.plot(probs, precisions_p, color="blue", linestyle="-", label="Precision P")
        ax.plot(probs, recalls_p, color="red", linestyle="-", label="Recall P")
        ax.plot(probs, f1_p, color="black", linestyle="-", label="F1 P")
    if metrics_s:
        ax.plot(probs, precisions_s, color="blue", linestyle="--", label="Precision S")
        ax.plot(probs, recalls_s, color="red", linestyle="--", label="Recall S")
        ax.plot(probs, f1_s, color="black", linestyle="--", label="F1 S")

    text_box = AnchoredText(s=f"F1 P: {f1_p_best['best_f1_p']:.2f}\n"
                              f"F1 S {f1_s_best['best_f1_s']:.2f}",
                            frameon=False,
                            loc="upper center",
                            pad=0.5)
    plt.setp(text_box.patch,
             facecolor='white',
             alpha=0.5)
    ax.add_artist(text_box)

    ax.set_ylim(0.25, 1.05)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pick threshold")
    ax.set_ylabel("Precision / Recall")
    ax.legend()
    ax.grid(visible=True)
    # ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"./metrics/probabilities_{parameters['filename']}.svg")


def test_on_catalog(model: seisbench.models.phasenet.PhaseNet,
                    catalog: obspy.Catalog,
                    station_json: str,
                    starttime: obspy.UTCDateTime,
                    endtime: obspy.UTCDateTime,
                    client: Union[FDSNClient, SDSClient],
                    residual: float = 0.3,
                    verbose: bool = False,
                    P_threshold: float = 0.2,
                    S_threshold: float = 0.2,
                    overlap: int = 2500,
                    blinding: list = [250, 250],
                    plot_waveforms: bool = False):
    """
    Testing a fully trained PhaseNet model on picks from a seismicity catalogue.
    """
    # Load stations from station_json
    stations = load_stations(station_json=station_json)

    # Gather all picks from catalogue
    all_picks = {}
    for event in catalog:
        picks = event_picks(event=event)
        for station_picks in picks.keys():
            for phase in picks[station_picks].keys():
                if station_picks not in all_picks.keys():
                    all_picks.update({station_picks: {"P": [], "S": []}})

                if phase.lower() in ["pg", "pn", "p"]:
                    phase_dict = "P"
                elif phase.lower() in ["sg", "sn", "s"]:
                    phase_dict = "S"
                all_picks[station_picks][phase_dict].append(picks[station_picks][phase])

    # Loop over each station
    f1_list_p = []
    f1_list_s = []
    precision_list_p = []
    precision_list_s = []
    recall_list_p = []
    recall_list_s = []
    for station_id in stations["id"]:
        network, station, location = station_id.split(".")

        # Test whether station_id is in all_picks, otherwise continue with next station
        if station_id not in all_picks.keys():
            continue

        # Load stream and predict picks
        stream = client.get_waveforms(station=station,
                                      network=network,
                                      location=location,
                                      channel="*",
                                      starttime=starttime,
                                      endtime=endtime)

        sb_picks = model.classify(stream,
                                  P_threshold=P_threshold,
                                  S_threshold=S_threshold,
                                  blinding=blinding,
                                  overlap=overlap)

        # Copy PhaseNet picks for plot
        sb_picks_copy = copy.deepcopy(sb_picks)

        # Number of detected picks by SeisBench
        total_p = 0
        total_s = 0
        for pick in sb_picks.picks:
            if pick.phase == "P":
                total_p += 1
            elif pick.phase == "S":
                total_s += 1

        # Detect true positive picks by looping over all picks from catalogue for station
        correct_p = 0
        correct_s = 0
        for phase, peak_times in all_picks[station_id].items():
            for peak_time in peak_times:  # loop over all arrival times in catalogue for phase
                for sbpick_idx, pick in enumerate(sb_picks.picks):
                    if np.abs(pick.peak_time - peak_time) <= residual:
                        if phase == "P" and pick.phase == "P":
                            correct_p += 1
                            print(peak_time)
                            # Remove pick from sb_picks.picks
                            sb_picks.picks.pop(sbpick_idx)
                        elif phase == "S" and pick.phase == "S":
                            correct_s += 1
                            # Remove pick from sb_picks.picks
                            sb_picks.picks.pop(sbpick_idx)

        # Calculate precision, recall and f1
        # True positives: Picks detected by SeisBench that are also in catalogue
        # False positives: Picks detected by SeisBench but not in catalogue (could be new picks)
        # False negatives: Number of picks that have been not picked by SeisBench
        tp_p = correct_p  # True positive P
        tp_s = correct_s  # True positive S
        fp_p = total_p - correct_p
        fp_s = total_s - correct_s
        fn_p = len(all_picks[station_id]["P"]) - tp_p
        fn_s = len(all_picks[station_id]["S"]) - tp_s

        # Check whether fp_p + fp_s = len(sb_picks.picks)
        if fp_p + fp_s != len(sb_picks.picks):
            raise ValueError("Something is wrong with the picks")


        # Plot stream, picks from catalogue and PhaseNet picks
        if plot_waveforms is True:
            fig = plt.figure(figsize=(15, 10))
            axs = fig.subplots(3, 1,
                               sharex=True,
                               gridspec_kw={"hspace": 0, "height_ratios": [1, 1, 1]})

            # Plot traces
            for ax, trace in zip(axs, stream):
                ax.plot_date(trace.times("matplotlib"), trace.data - np.mean(trace.data),
                             label=f"{trace.stats.station}.{trace.stats.channel}",
                             linestyle='solid', marker=None,
                             color="k", linewidth=0.5)
                ax.legend()

            # Plot catalogue picks
            for phase, peak_times in all_picks[station_id].items():
                color = "tab:blue" if phase == "P" else "tab:orange"
                for peak_time in peak_times:
                    for ax in axs:
                        ax.axvline(x=peak_time.matplotlib_date,
                                   color=color)

            # Plot PhaseNet picks
            for pick in sb_picks_copy.picks:
                color = "blue" if pick.phase == "P" else "red"
                for ax in axs:
                    ax.axvline(x=pick.peak_time.matplotlib_date,
                               color=color,
                               linestyle="--")
            # plt.suptitle(f"TPR-P: {tp_p / len(all_picks[station_id]['P'])} | "
            #              f"TPR-S: {tp_s/ len(all_picks[station_id]['S'])}")
            plt.suptitle(f"correct P {correct_p} | correct S {correct_s}")
            plt.show()

        precision_p = tp_p / (tp_p + fp_p + 1e-12)
        precision_s = tp_s / (tp_s + fp_s + 1e-12)
        recall_p = tp_p / (tp_p + fn_p + 1e-12)
        recall_s = tp_s / (tp_s + fn_s + 1e-12)
        f1_p = 2 * ((precision_p * recall_p) / (precision_p + recall_p + 1e-12))
        f1_s = 2 * ((precision_s * recall_s) / (precision_s + recall_s + 1e-12))

        precision_list_p.append(precision_p)
        precision_list_s.append(precision_s)
        recall_list_p.append(recall_p)
        recall_list_s.append(recall_s)
        f1_list_p.append(f1_p)
        f1_list_s.append(f1_s)

        if verbose is True:
            print(f"{station_id}\n"
                  f"Prec P: {precision_p:2f}  Rec P: {recall_p:2f}  F1 P: {f1_p:2f}\n"
                  f"Prec S: {precision_s:2f}  Rec S: {recall_s:2f}  F1 S: {f1_s:2f}\n")

    # Calculate average of each metric
    prec_p = np.average(precision_list_p)
    prec_s = np.average(precision_list_s)
    rec_p = np.average(recall_list_p)
    rec_s = np.average(recall_list_s)
    f1_p = np.average(f1_list_p)
    f1_s = np.average(f1_list_s)

    return prec_p, prec_s, rec_p, rec_s, f1_p, f1_s


def compare_models(models: dict, datasets: list, colors: Union[list, None] = None):
    """
    P is always solid, S is dashed line
    """
    # Setup parameters
    probs = np.linspace(0.00, 1.0, num=5)
    parameters = dict(
        sigma=10,
        nworkers=10,
        batch_size=512,
        win_len_factor=10,
        arrival_residual=25,
        nsamples=3001,
        sampling_rate=100,
        labeler="gaussian"
    )

    if not colors:
        colors = []
        for i in range(len(models)):
            colors.append('#%06X' % int(i*200))

    if len(colors) < len(models.keys()):
        msg = "Too many models or to less colors"
        raise ValueError(msg)

    # Read datasets
    for index, dataset in enumerate(datasets):
        if index == 0:
            seisbench_dataset = sbd.WaveformDataset(path=dataset,
                                                    sampling_rate=parameters["sampling_rate"])
        else:
            seisbench_dataset += sbd.WaveformDataset(path=dataset,
                                                     sampling_rate=parameters["sampling_rate"])

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Create figure canvas
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Loop over each model
    for index, (name, path) in enumerate(models.items()):
        # Load model
        try:
            model = torch.load(path, map_location=torch.device("cpu"))
        except:
            model = sbm.PhaseNet.from_pretrained(path)

        # Test model
        # Loop over each probability
        recalls_p, recalls_s = [], []
        for prob in probs:
            parameters["true_pick_prob"] = prob
            metrics_p, metrics_s = test_model(model=model, test_dataset=test, plot_residual_histogram=False,
                                              **parameters)
            recalls_p.append(metrics_p.recall)
            recalls_s.append(metrics_s.recall)

        # Plot precisions
        ax.plot(probs, recalls_p, color=colors[index], linestyle="-", linewidth=2, label=f"P {name}")
        ax.plot(probs, recalls_s, color=colors[index], linestyle="--", linewidth=2, label=f"S {name}")

    # Create legend and limits
    ax.set_ylim(0.25, 1.05)
    ax.set_xlabel("True pick probability")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(visible=True)
    plt.show()



if __name__ == "__main__":
    parfile = "/home/jheuel/code/train_phasenet/test_model.yml"
    # main(parfile)
    # probabilities(parfile)
    # misclassified_data(parfile)

    # models = glob.glob("/home/jheuel/code/train_phasenet/models/final_models/*.pth")[:2]
    # models_dct = {}
    # for model in models:
    #     models_dct.update({pathlib.Path(model).stem: model})
    #
    # compare_models(models=models_dct,
    #                datasets=glob.glob("/home/jheuel/scratch/ai_datasets/ps_filtered/*"))

    # parfiles = glob.glob("/home/jheuel/code/train_phasenet/models/final_models/*.yml")
    # for parfile in parfiles:
    #     probabilities(parfile=parfile,
    #                   probs=np.linspace(0, 1, 20),
    #                   model_path="/home/jheuel/code/train_phasenet/models/final_models")

    probabilities(parfile=parfile,
                  probs=np.linspace(1e-3, 1.0, 20),
                  model_path=None)