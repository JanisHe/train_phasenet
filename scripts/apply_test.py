import pathlib

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import seisbench.models as sbm
import seisbench.generate as sbg
from seisbench.util import worker_seeding
import matplotlib.pyplot as plt

from core.pn_utils import get_phase_dict, test_model, get_picks
from core.utils import is_nan, read_datasets


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

    picks_and_probs = get_picks(model=model, dataloader=test_loader, sigma=parameters["sigma"],
                                win_len_factor=parameters["win_len_factor"], return_data=True)
    
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


def probabilities(parfile):
    """
    Compute precision and recall for different probabilities
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

    # Test model on test data from dataset
    # Loop over true_pick_probabilities
    probs = np.linspace(0.05, 0.8, num=5)
    precisions_p, precisions_s = [],  []
    recalls_p, recalls_s = [], []
    tp_p, fp_p, fn_p = [], [], []
    tp_s, fp_s, fn_s = [], [], []
    for prob in probs:
        parameters["true_pick_prob"] = prob
        metrics_p, metrics_s = test_model(model=model, test_dataset=test, plot_residual_histogram=False, **parameters)
        precisions_p.append(metrics_p.precision)
        precisions_s.append(metrics_s.precision)
        recalls_p.append(metrics_p.recall)
        recalls_s.append(metrics_s.recall)

        tp_p.append(metrics_p.true_positive)
        fp_p.append(metrics_p.false_positive)
        fn_p.append(metrics_p.false_negative)
        tp_s.append(metrics_s.true_positive)
        fp_s.append(metrics_s.false_positive)
        fn_s.append(metrics_s.false_negative)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(probs, precisions_p, label="Precision P")
    ax.plot(probs, precisions_s, label="Precision S")
    ax.plot(probs, recalls_p, label="Recall P", linestyle="--")
    ax.plot(probs, recalls_s, label="Recall S", linestyle="--")
    ax.set_ylim(0.25, 1.05)
    ax.set_xlabel("True pick probability")
    ax.set_ylabel("Precision / Recall")
    ax.legend()
    ax.grid(visible=True)

    # ax2 = fig.add_subplot(122)
    # ax2.plot(probs, tp_p, label="TP P", color="b")
    # ax2.plot(probs, fp_p, label="FP P", color="r")
    # ax2.plot(probs, fn_p, label="FN P", color="g")
    # ax2.plot(probs, tp_s, label="TP S", linestyle="--", color="b")
    # ax2.plot(probs, fp_s, label="FP S", linestyle="--", color="r")
    # ax2.plot(probs, fn_s, label="FN S", linestyle="--", color="g")
    # ax2.set_ylim(0, len(test))
    # ax2.set_xlabel("True pick probability")
    # ax2.legend()
    # ax2.grid(visible=True)

    plt.savefig(f"./metrics/probabilities_{parameters['filename']}.svg")


if __name__ == "__main__":
    parfile = "/home/jheuel/code/train_phasenet/test_model.yml"
    # main(parfile)
    probabilities(parfile)
    # misclassified_data(parfile)
