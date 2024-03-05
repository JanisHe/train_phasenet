import sys
import os
import shutil
import pathlib
import yaml
import torch

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from pn_utils import get_phase_dict, test_model
from torch_functions import train_model, VectorCrossEntropyLoss


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
    shutil.copyfile(src=parfile, dst=f"./parfiles/{filename}.yml")

    # Read datasets
    for lst in parameters['datasets']:
        for dataset_count, dataset in enumerate(lst.values()):
            if dataset_count == 0:
                seisbench_dataset = sbd.WaveformDataset(path=Path(dataset),
                                                        sampling_rate=parameters["sampling_rate"])
            else:
                seisbench_dataset += sbd.WaveformDataset(path=Path(dataset),
                                                         sampling_rate=parameters["sampling_rate"])

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Define generators for training and validation
    train_generator = sbg.GenericGenerator(train)
    val_generator = sbg.GenericGenerator(validation)

    # Build augmentations and labels
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()), samples_before=parameters["nsamples"],
                               windowlen=2 * parameters["nsamples"], selection="random",
                               strategy="variable"),
        sbg.RandomWindow(windowlen=parameters["nsamples"], strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),  # Paper zhu: std and not peak
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=get_phase_dict(), sigma=parameters["sigma"], dim=0)
    ]

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

    # Load model
    if parameters["preload_model"]:
        model = sbm.PhaseNet.from_pretrained(parameters["preload_model"])
    else:
        model = sbm.PhaseNet(phases="PSN", norm="peak")

    # Start training
    # specify loss function
    loss_fn = VectorCrossEntropyLoss()

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

    model, train_loss, val_loss = train_model(model=model,
                                              patience=parameters["patience"],
                                              epochs=parameters["epochs"],
                                              loss_fn=loss_fn, optimizer=optimizer,
                                              train_loader=train_loader,
                                              validation_loader=val_loader)

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
    precission_p, precission_s, recall_p, recall_s, f1_p, f1_s = test_model(model=model, test_dataset=test,
                                                                            parameters=parameters,
                                                                            plot_residual_histogram=True)

    print("Precision P:", precission_p)
    print("Precision S:", precission_s)
    print("Recall P:", recall_p)
    print("Recall S:", recall_s)
    print("F1 P:", f1_p)
    print("F1 S:", f1_s)





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
