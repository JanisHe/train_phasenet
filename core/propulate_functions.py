import os
import torch
import time
import logging
import yaml
import seisbench # noqa
import socket
import pathlib

import numpy as np

import datetime as dt
import seisbench.data as sbd  # noqa
import seisbench.models as sbm # noqa
import seisbench.generate as sbg # noqa
from mpi4py import MPI
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding # noqa
import torch.utils.data.distributed as datadist
from sklearn.metrics import auc

from core.torch_functions import EarlyStopping, VectorCrossEntropyLoss, test_model
from core.utils import read_datasets, add_fake_events, get_phase_dict, check_parameters


log = logging.getLogger("propulate")  # Get logger instance.
SUBGROUP_COMM_METHOD = "nccl-slurm"
GPUS_PER_NODE = 4


def train_model_propulate(model,
                          train_loader,
                          validation_loader,
                          loss_fn,
                          optimizer=None,
                          epochs=50,
                          patience=5,
                          lr_scheduler=None,
                          trace_func=print):
    """

    """
    # Initialize lists to track losses
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    # Load optimizer
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize early stopping class
    early_stopping = EarlyStopping(patience=patience,
                                   verbose=False,
                                   path_checkpoint=None,
                                   trace_func=trace_func)

    # Loop over each epoch to start training
    rank = dist.get_rank()

    for epoch in range(epochs):
        # Train model (loop over each batch; batch_size is defined in DataLoader)
        train_loader.sampler.set_epoch(epoch)
        validation_loader.sampler.set_epoch(epoch)
        # pbar.set_description_str(desc=f"rank {rank} | epoch {epoch + 1}")
        trace_func(f"Rank {rank} in epoch {epoch + 1}")

        for batch_id, batch in enumerate(train_loader):
            # Compute prediction and loss
            try:
                pred = model(batch["X"].to(model.device))
            except RuntimeError:  # return empty lists for train and validation loss since parameters do mnot match
                return None, [], []
            loss = loss_fn(y_pred=pred, y_true=batch["y"].to(model.device))

            # Do backpropagation
            optimizer.zero_grad()  # clear the gradients of all optimized variables
            loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # perform a single optimization step (parameter update)

            # Compute loss for each batch and write loss to predefined lists
            dist.all_reduce(loss)
            loss /= dist.get_world_size()
            train_loss.append(loss.item())

        # Validate the model
        model.eval()  # Close the model for validation / evaluation
        trace_func(f"Validate rank {rank} for epoch {epoch + 1}")
        with torch.no_grad():  # Disable gradient calculation
            for batch in validation_loader:
                try:
                    pred = model(batch["X"].to(model.device))
                except RuntimeError:
                    continue
                val_loss = loss_fn(pred, batch["y"].to(model.device))
                dist.all_reduce(val_loss)
                val_loss /= dist.get_world_size()
                valid_loss.append(val_loss.item())

        # Determine average training and validation loss
        if len(train_loss) > 0 and len(valid_loss) > 0:
            avg_train_loss.append(sum(train_loss) / len(train_loss))
            avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

        # Re-open model for next epoch
        model.train()

        # Clear training and validation loss lists for next epoch
        train_loss = []
        valid_loss = []

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if len(avg_valid_loss) > 0:
            early_stopping(avg_valid_loss[-1], model)

        if early_stopping.early_stop:
            trace_func(f"Validation loss does not decrease further for rank {rank}. "
                       f"Early stopping!")
            break

    return model, avg_train_loss, avg_valid_loss


def torch_process_group_init_propulate(subgroup_comm: MPI.Comm,
                                       method: str,
                                       trace_func=print) -> None:
    """
    Create the torch process group of each multi-rank worker from a subgroup of the MPI world.

    Parameters
    ----------
    subgroup_comm : MPI.Comm
        The split communicator for the multi-rank worker's subgroup. This is provided to the individual's loss function
        by the ``Islands`` class if there are multiple ranks per worker.
    method : str
        The method to use to initialize the process group.
        Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
        If CUDA is not available, ``gloo`` is automatically chosen for the method.
    trace_func: prints output. Default is print statement
    """
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    comm_rank, comm_size = subgroup_comm.rank, subgroup_comm.size

    # Get master address and port
    # Don't want different groups to use the same port.
    subgroup_id = MPI.COMM_WORLD.rank // comm_size
    port = 29500 + subgroup_id

    if comm_size == 1:
        return
    master_address = f"{socket.gethostname()[:-7]}i"  # THIS IS THE NEW BIT! IT WILL PULL OUT THE rank-0 NODE NAME
    # master_address = f"{socket.gethostname()}"
    # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
    master_address = subgroup_comm.bcast(str(master_address), root=0)

    # Save environment variables.
    os.environ["MASTER_ADDR"] = master_address
    # Use the default PyTorch port.
    os.environ["MASTER_PORT"] = str(port)

    if not torch.cuda.is_available():
        method = "gloo"
        trace_func("No CUDA devices found: Falling back to gloo.")
    else:
        trace_func(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        num_cuda_devices = torch.cuda.device_count()
        device_number = MPI.COMM_WORLD.rank % num_cuda_devices
        trace_func(f"device count: {num_cuda_devices}, device number: {device_number}")
        torch.cuda.set_device(device_number)

    time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
    if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
        dist.init_process_group(
            backend="nccl",
            rank=comm_rank,
            world_size=comm_size,
        )

    elif method == "nccl-slurm":  # Use NCCL with a TCP store.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    elif method == "gloo":  # Use gloo.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    else:
        raise NotImplementedError(
            f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]!"
        )

    # Call a barrier here in order for sharp to use the default comm.
    if dist.is_initialized():
        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == comm_size, "Failed test of dist!"
    else:
        disttest = None
    trace_func(
        f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}"
    )


def get_data_loaders(comm: MPI.Comm,
                     parameters: dict,
                     model,
                     trace_func: print):

    # Read waveform datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      component_order=model.component_order,
                                      dataset_key="datasets",
                                      filter=parameters.get("filter"),
                                      trace_func=trace_func)

    # Add fake events to metadata
    if parameters.get("add_fake_events"):
        add_fake_events(sb_dataset=seisbench_dataset,
                        percentage=parameters["add_fake_events"])

    # Split dataset in train, dev (validation) and test
    # TODO: Test dataset is not necessary for propulate
    train, validation, test = seisbench_dataset.train_dev_test()

    # Define generators for training and validation
    train_generator = sbg.GenericGenerator(train)
    val_generator = sbg.GenericGenerator(validation)

    # Build augmentations and labels
    # Ensure that all phases are in requested window
    augmentations = [
        sbg.WindowAroundSample(list(get_phase_dict(p_phase=parameters["p_phase"],
                                                   s_phase=parameters["s_phase"]).keys()),
                               samples_before=int(0.8 * parameters["nsamples"]),
                               windowlen=int(1.5 * parameters["nsamples"]),
                               selection="first",
                               strategy="variable"),
        sbg.RandomWindow(windowlen=parameters["nsamples"],
                         strategy="pad"),
        sbg.ProbabilisticLabeller(shape=parameters["labeler"],
                                  label_columns=get_phase_dict(p_phase=parameters["p_phase"],
                                                               s_phase=parameters["s_phase"]),
                                  sigma=parameters["sigma"],
                                  dim=0,
                                  model_labels=model.labels,
                                  noise_column=True)
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
    augmentations.append(sbg.Normalize(demean_axis=-1,
                                       amp_norm_axis=-1,
                                       amp_norm_type=model.norm))
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


def ind_loss(h_params: dict[str, int | float],
             subgroup_comm: MPI.Comm) -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    h_params : dict[str, int | float]
        The hyperparameters to be optimized evolutionarily. Here, batch_size, learning_rate,
        stride, in_samples, kernel_size, filters_root, drop_rate, depth
    subgroup_comm : MPI.Comm
        Each multi-rank worker's subgroup communicator.

    Returns
    -------
    float
        The trained model's validation loss.
    """
    torch_process_group_init_propulate(subgroup_comm,
                                       method=SUBGROUP_COMM_METHOD,
                                       trace_func=log.info)

    # Read PhaseNet parameters from parfile
    with open(h_params["parfile"], "r") as file:
        parameters = yaml.safe_load(file)

    # Extract hyperparameter combination to test from input dictionary and add to parameters dictionary
    # If parameter is not given, then default value is used
    parameters["learning_rate"] = h_params["learning_rate"]
    parameters["batch_size"] = h_params["batch_size"]
    parameters["nsamples"] = h_params["nsamples"]
    parameters["stride"] = h_params["stride"]
    parameters["kernel_size"] = h_params["kernel_size"]
    parameters["filter_factor"] = h_params["filter_factor"]
    parameters["filters_root"] = h_params["filters_root"]
    parameters["depth"] = h_params["depth"]
    parameters["drop_rate"] = h_params["drop_rate"]
    parameters["activation_function"] = h_params["activation_function"]
    parameters["loss_fn"] = h_params["loss_fn"]

    # Select correct activation function
    activation_function = h_params["activation_function"]
    if activation_function.lower() == "elu":
        activation_function = torch.nn.ELU()
    elif activation_function.lower() == "relu":
        activation_function = torch.nn.ReLU()
    elif activation_function.lower() == "gelu":
        activation_function = torch.nn.GELU()
    elif activation_function.lower() == "leakyrelu":
        activation_function = torch.nn.LeakyReLU()
    else:
        msg = f"The activation function {activation_function} is not implemented."
        raise ValueError(msg)

    # print loaded parameters
    # rank = dist.get_rank()
    # print_params = copy.copy(parameters)
    # print_params.pop("datasets")
    # log.info(msg=f"rank: {rank} | {print_params}")

    # Check parameters and modify e.g. metadata or if key is not found, default value is used
    parameters = check_parameters(parameters=parameters)

    # Set number of workers for PyTorch
    # https://github.com/pytorch/pytorch/issues/101850
    os.sched_setaffinity(0, range(os.cpu_count()))

    # Load model
    model = sbm.VariableLengthPhaseNet(phases=parameters["phases"],
                                       in_samples=parameters["nsamples"],
                                       classes=len(parameters["phases"]),
                                       norm="peak",
                                       stride=parameters["stride"],
                                       filter_factor=parameters["filter_factor"],
                                       kernel_size=parameters["kernel_size"],
                                       filters_root=parameters["filters_root"],
                                       depth=parameters["depth"],
                                       drop_rate=parameters["drop_rate"],
                                       activation=activation_function)

    train_loader, val_loader, test = get_data_loaders(comm=subgroup_comm,
                                                      parameters=parameters,
                                                      model=model,
                                                      trace_func=log.info)

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        # device = subgroup_comm.rank % GPUS_PER_NODE   # Au Haicore werden alle gesehen, deswegen wird device gewaehlt
        device = "cuda"   # Auf Juwels muss als device "cuda" benutzt werden
    else:
        device = "cpu"

    model = model.to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)  # Wrap model with DDP.

    # Start training
    # specify loss function
    loss_fn = VectorCrossEntropyLoss()

    # specify learning rate and optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=parameters["learning_rate"])


    # Train loop returns None, [], [] if model parameters to not fit, i.e. the model with the given parameters
    # is not set up correctly.
    model, train_loss, val_loss = train_model_propulate(model=model,
                                                        patience=parameters.get("patience"),
                                                        epochs=parameters["epochs"],
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer,
                                                        train_loader=train_loader,
                                                        validation_loader=val_loader,
                                                        lr_scheduler=None,
                                                        trace_func=log.info)

    # Return best validation loss as an individual's loss (trained so lower is better).
    dist.barrier()
    dist.destroy_process_group()

    # Instead of return the average loss value, the model is evaluated and precision, recall and
    # F1-score are determined for different probabilities
    # Add parameters for testing each model
    parameters["arrival_residual"] = 30
    parameters["win_len_factor"] = 10

    # Only test and save model for rank 0 since gradients are synchronized in backward passes
    # TODO: Probably something still does not work with unwrapping the model
    if model:
        log.info("Testing model on test dataset")
        probs = np.linspace(start=1e-3,
                            stop=1,
                            num=20)
        precision_p, precision_s = np.zeros(len(probs)), np.zeros(len(probs))
        recalls_p, recalls_s = np.zeros(len(probs)), np.zeros(len(probs))
        f1_p, f1_s = np.zeros(len(probs)), np.zeros(len(probs))
        for index, prob in enumerate(probs):
            parameters["true_pick_prob"] = prob

            metrics_p, metrics_s = test_model(model=model.module,  # Since model is wrapped with DDP
                                              test_dataset=test,
                                              **parameters)

            if metrics_p:
                precision_p[index] = metrics_p.precision
                recalls_p[index] = metrics_p.recall
                f1_p[index] = metrics_p.f1_score
            if metrics_s:
                precision_s[index] = metrics_s.precision
                recalls_s[index] = metrics_s.recall
                f1_s[index] = metrics_s.f1_score

        # Determining best F1 score from netrics
        if metrics_p:
            f1_p_best = {"best_f1_p": np.max(f1_p),
                         "idx": np.argmax(f1_p)}
        if metrics_s:
            f1_s_best = {"best_f1_s": np.max(f1_s),
                         "idx": np.argmax(f1_s)}

        # test whether index of best f1 scores for P and S is greater than 0
        # Usually, if highest f1-score is close to a probability of zero, the best threshold for
        # P and S is close to zero
        if metrics_p and metrics_s:  # Overall average F1 score for P and S
            if f1_p_best["idx"] > 0 and f1_s_best["idx"] > 0:
                avg_auc = 1 - np.average(a=[f1_p_best["best_f1_p"],
                                            f1_s_best["best_f1_s"]])
            else:
                avg_auc = 1
        elif metrics_p and metrics_s is None:  # Single model for P phase
            if f1_p_best["idx"] > 0:
                avg_auc = 1 - f1_p_best["best_f1_p"]
            else:
                avg_auc = 1
        elif metrics_p is None and metrics_s:  # Single model for S phase
            if f1_s_best["idx"] > 0:
                avg_auc = 1 - f1_s_best["best_f1_s"]
            else:
                avg_auc = 1

        # Determine area under precision-recall curve
        # If recall does not increasing or decreasing monotonically, then the model is not validated further and
        # the average AUCPR is set to 1000, which is the value propulate optimizes for.
        # The following code block computes the area und the precision-recall curve to let propulate optimize on
        # this value
        try:
            if metrics_p:
                auc_p = auc(x=recalls_p,
                            y=precision_p)
            if metrics_s:
                auc_s = auc(x=recalls_s,
                            y=precision_s)
        except ValueError:   # recall is not monotonic increasing or monotonic decreasing
            avg_auc = 1
    else:
        avg_auc = 1

    # Save model if avg_auc is not 1000
    if avg_auc < 1:
        filename = f"{pathlib.Path(h_params['parfile']).stem}_{avg_auc:.5f}.pt"
        try:
            if os.path.isfile(path=os.path.join(parameters["checkpoint_path"], "models")) is False:
                os.makedirs(os.path.join(parameters["checkpoint_path"], "models"))
        except FileExistsError:
            pass
        torch.save(obj=model.module,   # Unwrap DDP model
                   f=os.path.join(parameters["checkpoint_path"], "models", filename))

        # Write out parameters for successull tested model
        with open(os.path.join(parameters["checkpoint_path"], "tested_models"), "a") as f:
            f.write("##################################\n")
            for key, item in parameters.items():
                f.write(f"{key}: {item}\n")
            f.write(f"Avg. AUCPR: {avg_auc:.5f}\n")
            f.write(f'model_filename: {os.path.join(parameters["checkpoint_path"], "models", filename)}\n')
            f.write("##################################\n")
    else:  # Write parameters into file, where not fitting model parameters are stored
        with open(os.path.join(parameters["checkpoint_path"], "failed_models"), "a") as f:
            f.write("##################################\n")
            for key, item in parameters.items():
                f.write(f"{key}: {item}\n")
            f.write("##################################\n")

    return avg_auc
