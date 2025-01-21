import os
import pathlib
import tqdm

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import seisbench.models as sbm  # noqa
import seisbench.generate as sbg # noqa
import seisbench.data as sbd  # noqa
from seisbench.util import worker_seeding # noqa
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from typing import Union
from sklearn.metrics import auc

from core.utils import is_nan, read_datasets, get_phase_dict, get_picks, best_threshold
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
            precisions_p.append(metrics_p.precision)
            precisions_s.append(metrics_s.precision)
            recalls_p.append(metrics_p.recall)
            recalls_s.append(metrics_s.recall)
            f1_p.append(metrics_p.f1_score)
            f1_s.append(metrics_s.f1_score)

            tp_p.append(metrics_p.true_positive)
            fp_p.append(metrics_p.false_positive)
            fn_p.append(metrics_p.false_negative)
            tp_s.append(metrics_s.true_positive)
            fp_s.append(metrics_s.false_positive)
            fn_s.append(metrics_s.false_negative)

            pbar.update()

    # Finding the best thresholds for P and S from precision-recall curve
    # Determining distances of each precision-recall pair to point (1, 1)
    # Smallest distance represents the best threshold
    rp_p = np.array(list(zip(recalls_p, precisions_p)))  # recall-precision array for P
    rp_s = np.array(list(zip(recalls_s, precisions_s)))  # recall-precision array for S
    best_threshold_p = best_threshold(recall_precision_array=rp_p,
                                      thresholds=probs)
    best_threshold_s = best_threshold(recall_precision_array=rp_s,
                                      thresholds=probs)

    # Determining area under precision-recall curve
    try:
        auc_p = auc(x=recalls_p,
                    y=precisions_p)
    except ValueError:
        auc_p = 999
    try:
        auc_s = auc(x=recalls_s,
                    y=precisions_s)
    except ValueError:
        auc_s = 999

    # Plot
    fig= plt.figure(figsize=(11, 5))
    ax_pr = fig.add_subplot(121)
    ax_pr.plot(recalls_p, precisions_p, label=f"P (AUC: {auc_p:.2f}")
    ax_pr.plot(recalls_s, precisions_s, label=f"S (AUC: {auc_s:.2f}")
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
    ax.plot(probs, precisions_p, color="blue", linestyle="-", label="Precision P")
    ax.plot(probs, precisions_s, color="blue", linestyle="--", label="Precision S")
    ax.plot(probs, recalls_p, color="red", linestyle="-", label="Recall P")
    ax.plot(probs, recalls_s, color="red", linestyle="--", label="Recall S")
    ax.plot(probs, f1_p, color="black", linestyle="-", label="F1 P")
    ax.plot(probs, f1_s, color="black", linestyle="--", label="F1 S")
    ax.set_ylim(0.25, 1.05)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pick threshold")
    ax.set_ylabel("Precision / Recall")
    ax.legend()
    ax.grid(visible=True)
    # ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"./metrics/probabilities_{parameters['filename']}.svg")


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