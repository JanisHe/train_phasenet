import sys
import os
import shutil
import pathlib
import yaml
import torch
import requests

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
import torch.optim as optim

from pn_utils import get_phase_dict, test_model
from torch_functions import train_model, VectorCrossEntropyLoss
from utils import check_parameters, read_datasets


def main(parfile):
    """

    """
    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    filename = pathlib.Path(parameters["model_name"]).stem
    parameters["filename"] = filename

    # Make copy of parfile and rename it by filename given in parameters
    if not os.path.exists("./parfiles"):
        os.makedirs("./parfiles")
    try:
        shutil.copyfile(src=parfile, dst=f"./parfiles/{filename}.yml")
    except shutil.SameFileError:
        pass

    # Check parameters and modify e.g. metadata
    parameters = check_parameters(parameters=parameters)

    # Read waveform datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      dataset_key="datasets",
                                      filter=parameters.get("filter"))

    # Load noise_dataset data set
    if parameters.get("noise_datasets"):
        noise_dataset = read_datasets(parameters=parameters, dataset_key="noise_datasets")

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Copy tmp_metadata back to metadata and delete tmp_metadata.csv
    if parameters.get("add_fake_events"):
        for lst in parameters['datasets']:
            for dataset in lst.values():
                # Copy metadata file
                shutil.copyfile(src=os.path.join(dataset, "tmp_metadata.csv"),
                                dst=os.path.join(dataset, "metadata.csv"))
                os.remove(path=os.path.join(dataset, "tmp_metadata.csv"))

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

    # Move model to GPU if GPU is available
    if torch.cuda.is_available() is True:
        model.cuda()
        print("Running PhaseNet training on GPU.")

    # Define generators for training and validation
    train_generator = sbg.GenericGenerator(train)
    val_generator = sbg.GenericGenerator(validation)

    # Build augmentations and labels
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()), samples_before=parameters["nsamples"],
                               windowlen=2 * parameters["nsamples"], selection="random",
                               strategy="variable"),
        sbg.RandomWindow(windowlen=parameters["nsamples"], strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=model.norm),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(shape=parameters["labeler"],
                                  label_columns=get_phase_dict(), sigma=parameters["sigma"],
                                  dim=0, model_labels=model.labels, noise_column=True)
    ]

    if parameters.get("rotate") is True:
        augmentations.append(sbg.RotateHorizontalComponents())

    # Add RealNoise to augmentations if noise_datasets are in parmeters
    if parameters.get("noise_datasets"):
        augmentations.append(sbg.RealNoise(noise_dataset=noise_dataset,
                                           metadata_thresholds=dict(
                                               trace_Z_snr_db=10
                                           )))

    # Add augmentations to generators
    train_generator.add_augmentations(augmentations=augmentations)
    val_generator.add_augmentations(augmentations=augmentations)

    # Define generators to load data
    train_loader = DataLoader(dataset=train_generator, batch_size=parameters["batch_size"],
                              shuffle=True, num_workers=parameters["nworkers"],
                              worker_init_fn=worker_seeding)
    val_loader = DataLoader(dataset=val_generator, batch_size=parameters["batch_size"],
                            shuffle=False, num_workers=parameters["nworkers"],
                            worker_init_fn=worker_seeding)

    # Start training
    # specify loss function
    loss_fn = VectorCrossEntropyLoss()

    # specify learning rate and optimizer
    if isinstance(parameters["learning_rate"], float) or isinstance(parameters["learning_rate"], int):
        lr = parameters["learning_rate"]
    elif isinstance(parameters["learning_rate"], dict):
        lr = parameters["learning_rate"]["initial_lr"]
    else:
        raise ValueError("learning rate is not defined.")
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
                                              loss_fn=loss_fn, optimizer=optimizer,
                                              train_loader=train_loader,
                                              validation_loader=val_loader,
                                              lr_scheduler=scheduler)

    # Save model
    head, tail = os.path.split(parameters["model_name"])
    if head == "":
        if os.path.isdir(os.path.join(".", "models")) is False:
            os.makedirs(os.path.join(".", "models"))
        torch.save(model, os.path.join('.', 'models', parameters['model_name']))
    else:
        if os.path.isdir(head) is False:
            os.makedirs(head)
        torch.save(model, parameters['model_name'])
    print(f"Saved model as {parameters['model_name']}.")

    # Test model if test data are available
    if len(test) > 0:
        # Plot training and validation loss
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(train_loss)), train_loss, label="train")
        ax.plot(np.arange(len(val_loss)), val_loss, label="test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # Save figure
        if os.path.isdir(os.path.join(".", "loss_figures")) is False:
            os.makedirs(os.path.join(".", "loss_figures"))
        plt.savefig(os.path.join(".", "loss_figures", f"{filename}.png"))

        # Test model on test data from dataset
        metrics_p, metrics_s = test_model(model=model, test_dataset=test, plot_residual_histogram=True, **parameters)

        print("Precision P:", metrics_p.precision)
        print("Precision S:", metrics_s.precision)
        print("Recall P:", metrics_p.recall)
        print("Recall S:", metrics_s.recall)
        print("F1 P:", metrics_p.f1_score)
        print("F1 S:", metrics_s.f1_score)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        # msg = "No parfile is defined!"
        # raise FileNotFoundError(msg)
        parfile = "./pn_parfile.yml"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Start to train PhaseNet
    main(parfile=parfile)
