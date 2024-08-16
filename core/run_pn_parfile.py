import sys
import os
import shutil
import pathlib
import yaml
import torch
import requests
import tqdm

from mpi4py import MPI
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
import torch.utils.data.distributed as datadist
from torch.utils.hipify.hipify_python import meta_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pn_utils import get_phase_dict, test_model
from torch_functions import train_model, VectorCrossEntropyLoss
from utils import check_parameters, read_datasets, add_fake_events_to_metadata, torch_process_group_init


GPUS_PER_NODE = 4


def get_data_loaders(comm: MPI.Comm, parameters, model):
    # Read waveform datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      component_order=model.component_order,
                                      dataset_key="datasets",
                                      filter=parameters.get("filter"))

    # Add fake events to metadata
    if parameters.get("add_fake_events"):
        add_fake_events_to_metadata(sb_dataset=seisbench_dataset,
                                    number=int(len(seisbench_dataset) * parameters["add_fake_events"] / 100))

    # Split dataset in train, dev (validation) and test
    train, validation, test = seisbench_dataset.train_dev_test()

    # Define generators for training and validation
    train_generator = sbg.GenericGenerator(train)
    val_generator = sbg.GenericGenerator(validation)

    # Build augmentations and labels
    # Ensure that all phases are in requested window
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict().keys()),
                               samples_before=int(0.8 * parameters["nsamples"]),
                               windowlen=int(1.5 * parameters["nsamples"]),
                               selection="first",
                               strategy="variable"),
        sbg.RandomWindow(windowlen=parameters["nsamples"],
                         strategy="pad"),
        sbg.ProbabilisticLabeller(shape=parameters["labeler"],
                                  label_columns=get_phase_dict(), sigma=parameters["sigma"],
                                  dim=0, model_labels=model.labels, noise_column=True)
    ]

    if parameters.get("rotate") is True:
        augmentations.append(sbg.RotateHorizontalComponents())

    # Add RealNoise to augmentations if noise_datasets are in parmeters
    if parameters.get("noise_datasets"):
        noise_dataset = read_datasets(parameters=parameters, dataset_key="noise_datasets")
        # TODO: trace_Z_snr is hard coded
        augmentations.append(
            sbg.OneOf(
                augmentations=[sbg.RealNoise(
                    noise_dataset=noise_dataset,
                    metadata_thresholds={"trace_Z_snr_db": 10}
                )],
                probabilities=[0.5]
            )
        )

    # Change dtype of data (necessary for PyTorch and the last augmentation step)
    augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=model.norm))
    augmentations.append(sbg.ChangeDtype(np.float32))

    # Add augmentations to generators
    train_generator.add_augmentations(augmentations=augmentations)
    val_generator.add_augmentations(augmentations=augmentations)

    if comm.size > 1:  # Make the samplers use the torch world to distribute data
        train_sampler = datadist.DistributedSampler(train_generator,
                                                    seed=42)
        val_sampler = datadist.DistributedSampler(val_generator,
                                                  seed=42)
    else:
        train_sampler = None
        val_sampler = None

    num_workers = parameters["nworkers"]
    print(f"Use {num_workers} workers in dataloader.")

    # Define generators to load data
    train_loader = DataLoader(dataset=train_generator,
                              batch_size=parameters["batch_size"],
                              num_workers=parameters["nworkers"],
                              pin_memory=True,
                              persistent_workers=True,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler)

    val_loader = DataLoader(dataset=val_generator,
                            batch_size=parameters["batch_size"],
                            num_workers=parameters["nworkers"],
                            pin_memory=True,
                            persistent_workers=True,
                            shuffle=(val_sampler is None),
                            sampler=val_sampler)

    return train_loader, val_loader, test


def main(parfile):
    """

    """
    comm = MPI.COMM_WORLD
    torch_process_group_init(comm=comm, method="nccl-slurm")

    with open(parfile, "r") as file:
        parameters = yaml.safe_load(file)

    filename = pathlib.Path(parameters["model_name"]).stem
    parameters["filename"] = filename

    # Set number of workers for PyTorch
    # https://github.com/pytorch/pytorch/issues/101850
    os.sched_setaffinity(0, range(os.cpu_count()))

    # Make copy of parfile and rename it by filename given in parameters
    if not os.path.exists("./parfiles"):
        os.makedirs("./parfiles")
    try:
        shutil.copyfile(src=parfile, dst=f"./parfiles/{filename}.yml")
    except shutil.SameFileError:
        pass

    # Check parameters and modify e.g. metadata
    parameters = check_parameters(parameters=parameters)

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

    train_loader, val_loader, test = get_data_loaders(comm=comm,
                                                      parameters=parameters,
                                                      model=model)

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        device = comm.rank % GPUS_PER_NODE
    else:
        device = "cpu"

    model = model.to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)  # Wrap model with DDP.

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
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
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

    # Plot training and validation loss
    fig_loss = plt.figure()
    ax = fig_loss.add_subplot(111)
    ax.plot(np.arange(len(train_loss)), train_loss, label="train")
    ax.plot(np.arange(len(val_loss)), val_loss, label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # Save figure
    if os.path.isdir(os.path.join(".", "loss_figures")) is False:
        os.makedirs(os.path.join(".", "loss_figures"))
    fig_loss.savefig(os.path.join(".", "loss_figures", f"{filename}.png"))

    # Test model if test data are available
    if len(test) > 0:
        # Test model on test data from dataset for different probabilities
        precision_p, precision_s = [], []
        recalls_p, recalls_s = [], []
        f1_p, f1_s = [], []
        probs = np.linspace(0, 1, 20)
        with tqdm.tqdm(total=len(probs), desc=f"Testing model", ncols=100,
                       bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]") as pbar:
            for prob in probs:
                parameters["true_pick_prob"] = prob
                metrics_p, metrics_s = test_model(model=model, test_dataset=test, plot_residual_histogram=False,
                                                  **parameters)
                precision_p.append(metrics_p.precision)
                precision_s.append(metrics_s.precision)
                recalls_p.append(metrics_p.recall)
                recalls_s.append(metrics_s.recall)
                f1_p.append(metrics_p.f1_score)
                f1_s.append(metrics_s.f1_score)

                pbar.update()

        # Plot metrics for P and S in one figure
        fig_metrics = plt.figure()
        ax = fig_metrics.add_subplot(111)
        ax.plot(probs, precision_p, color="blue", linestyle="-", label="Precision P")
        ax.plot(probs, precision_s, color="blue", linestyle="--", label="Precision S")
        ax.plot(probs, recalls_p, color="red", linestyle="-", label="Recall P")
        ax.plot(probs, recalls_s, color="red", linestyle="--", label="Recall S")
        ax.plot(probs, f1_p, color="black", linestyle="-", label="F1 P")
        ax.plot(probs, f1_s, color="black", linestyle="--", label="F1 S")
        ax.set_xlabel("True pick probability")
        ax.grid(visible=True)
        ax.legend()

        if os.path.isdir(os.path.join(".", "metrics")) is False:
            os.makedirs(os.path.join(".", "metrics"))

        fig_metrics.savefig(fname=os.path.join(".", "metrics", f"{filename}.png"))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        parfile = "../pn_parfile.yml"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Start to train PhaseNet
    main(parfile=parfile)
