import os
import torch
import time
import logging
import yaml
import seisbench # noqa
import socket
import copy
# import torchvision

import numpy as np

import datetime as dt
import seisbench.data as sbd  # noqa
import seisbench.models as sbm # noqa
import seisbench.generate as sbg # noqa
from tqdm import tqdm
from mpi4py import MPI
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding # noqa
from tqdm.auto import tqdm
import torch.utils.data.distributed as datadist

from core.utils import read_datasets, add_fake_events, get_phase_dict, check_parameters, is_nan


log = logging.getLogger("propulate")  # Get logger instance.
SUBGROUP_COMM_METHOD = "nccl-slurm"
GPUS_PER_NODE = 4


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7,
                 verbose=False,
                 delta=0,
                 path_checkpoint=None,
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path_checkpoint (str, None): Path for the checkpoint to be saved to. If not None chechpoints are saved.
                            Default: None
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path_checkpoint
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')

        if self.path:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Metrics:
    """

    """
    def __init__(self,
                 predictions: list,
                 true_pick_prob: float=0.5,
                 arrival_residual: int=10):

        self.predictions = predictions
        self.true_pick_prob = true_pick_prob
        self.arrival_residual = arrival_residual

        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

        self.true_false_positives()

    def __str__(self):
        pass

    def true_false_positives(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        for prediction in self.predictions:
            if isinstance(prediction, list) is True:
                if len(prediction) == 0:
                    self.false_negative += 1
                else:
                    for element in prediction:
                        if element["peak_value"] >= self.true_pick_prob and abs(
                                element["residual"]) <= self.arrival_residual:
                            self.true_positive += 1
                        elif element["peak_value"] >= self.true_pick_prob and abs(
                            element["residual"]) > self.arrival_residual:
                            self.false_positive += 1
                        elif element["peak_value"] < self.true_pick_prob or abs(
                            element["residual"]) > self.arrival_residual:
                            self.false_negative += 1
            else:
                self.false_negative += 1

    @property
    def precision(self) -> float:
        try:
            return self.true_positive / (self.true_positive + self.false_positive)
        except ZeroDivisionError:
            # Limit of precision for high probabilities, e.g. 1
            # Avoiding division by zero
            return 1

    @property
    def recall(self) -> float:
        try:
            return self.true_positive / (self.true_positive + self.false_negative)
        except ZeroDivisionError:
            return 0

    @property
    def f1_score(self) -> float:
        return 2 * ((self.precision * self.recall) / (self.precision + self.recall))


class VectorCrossEntropyLoss:
    """
    Vector cross entropy as definded in Zhu & Beroza (2018).

    H(p, q) = \sum_{i=-1}^{3} \sum_x p_i(x) * log(q_i(x))
    p: true probabiliyt distribution
    q: predicted distribution
    """
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true, eps=1e-10):
        """
        :param y_pred:
        :param y_true:
        :param eps:
        """
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(-1)                 # Mean along sample dimension and sum along pick dimension
        h = h.mean()                           # Mean over batch axis

        return -h


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """

        """
        mse = (y_true - y_pred) ** 2
        mse = mse.mean(-1).sum(-1)
        mse = mse.mean()

        return mse


# class FocalLoss:
#     def __init__(self,
#                  alpha: float = 0.25,
#                  gamma: float = 2,
#                  reduction: str = "mean"):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def __call__(self,
#                  y_pred,
#                  y_true):
#         val =  torchvision.ops.sigmoid_focal_loss(inputs=y_pred,
#                                                   targets=y_true,
#                                                   alpha=self.alpha,
#                                                   gamma=self.gamma,
#                                                   reduction=self.reduction)
#
#         if self.reduction == "none":
#             val = val.mean(-1).sum(-1)
#             val = val.mean()
#
#         return val


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.

    https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """

    def __init__(
            self,
            best_valid_loss=float('inf'),
            model_name: str = "best_model.pth",
            verbose: bool = False,
            trace_func=print
    ):
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
        self.verbose = verbose
        self.trace_func = trace_func

    def __call__(
            self,
            current_valid_loss,
            epoch,
            model
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if self.verbose is True:
                self.trace_func(f"Saving best model for epoch {epoch + 1} as {self.model_name}")
            torch.save(obj=model,
                       f=self.model_name)


def train_model(model,
                train_loader,
                validation_loader,
                loss_fn,
                optimizer=None,
                epochs=50,
                patience=5,
                lr_scheduler=None,
                model_name: str = "my_best_model.pth",
                verbose: bool = True):
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
    early_stopping = EarlyStopping(patience=patience, verbose=False, path_checkpoint=None)

    # Initialize best model
    best_model = SaveBestModel(model_name=model_name)

    # Loop over each epoch to start training
    for epoch in range(epochs):
        # Train model (loop over each batch; batch_size is defined in DataLoader)
        # TODO (idea): test model with validation (compute metrics)
        num_batches = len(train_loader)
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}", ncols=100,
                  bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]") as pbar:
            for batch_id, batch in enumerate(train_loader):
                # Compute prediction and loss
                pred = model(batch["X"].to(model.device))
                loss = loss_fn(y_pred=pred, y_true=batch["y"].to(model.device))

                # Do backpropagation
                optimizer.zero_grad()   # clear the gradients of all optimized variables
                loss.backward()         # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()        # perform a single optimization step (parameter update)

                # Compute loss for each batch and write loss to predefined lists
                train_loss.append(loss.item())

                # Update progressbar
                pbar.set_postfix({"loss": str(np.round(loss.item(), 4))})
                pbar.update()

            # Validate the model
            model.eval()     # Close the model for validation / evaluation
            with torch.no_grad():   # Disable gradient calculation
                for batch in validation_loader:
                    pred = model(batch["X"].to(model.device))
                    valid_loss.append(loss_fn(pred, batch["y"].to(model.device)).item())

            # Determine average training and validation loss
            avg_train_loss.append(sum(train_loss) / len(train_loss))
            avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

            # Update progressbar
            pbar.set_postfix(
                {"loss": str(np.round(avg_train_loss[-1], 4)),
                 "val_loss": str(np.round(avg_valid_loss[-1], 4))}
            )

            # Save model if validation loss decreased
            best_model(current_valid_loss=avg_valid_loss[-1],
                       epoch=epoch,
                       model=model)

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
        early_stopping(avg_valid_loss[-1], model)

        if early_stopping.early_stop:
            if verbose:
                print("Validation loss does not decrease further. Early stopping")
            break

    if verbose is True:
        print(f"Saved model as {model_name}.")

    return model, avg_train_loss, avg_valid_loss


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
    # progress_bar = tqdm(total=epochs,
    #                     desc=f"rank {rank} | epoch   ",
    #                     ncols=100,
    #                     bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]",
    #                     position=rank)

    # with progress_bar as pbar:
    for epoch in range(epochs):
        # Train model (loop over each batch; batch_size is defined in DataLoader)
        train_loader.sampler.set_epoch(epoch)
        validation_loader.sampler.set_epoch(epoch)
        # pbar.set_description_str(desc=f"rank {rank} | epoch {epoch + 1}")
        # trace_func(f"Rank {rank} in epoch {epoch + 1}")

        for batch_id, batch in enumerate(train_loader):
            # Compute prediction and loss
            try:
                pred = model(batch["X"].to(model.device))
            except RuntimeError:  # return empty lists for train and validation loss since parameters do mnot match
                return [], []
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
        # trace_func(f"Validate rank {rank} for epoch {epoch + 1}")
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
            # trace_func(f"Validation loss does not decrease further for rank {rank}. "
            #            f"Early stopping!")
            break

    # pbar.close()

    return avg_train_loss, avg_valid_loss


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
    # master_address = f"{socket.gethostname()[:-7]}i"  # THIS IS THE NEW BIT! IT WILL PULL OUT THE rank-0 NODE NAME
    master_address = f"{socket.gethostname()}"
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
            timeout=dt.timedelta(seconds=60),
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
            timeout=dt.timedelta(seconds=60),
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
                     model):

    # Read waveform datasets
    seisbench_dataset = read_datasets(parameters=parameters,
                                      component_order=model.component_order,
                                      dataset_key="datasets",
                                      filter=parameters.get("filter"))

    # Add fake events to metadata
    if parameters.get("add_fake_events"):
        add_fake_events(sb_dataset=seisbench_dataset,
                        percentage=parameters["add_fake_events"])

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
                                  label_columns=get_phase_dict(),
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

    # Set number of workers for PyTorch
    # https://github.com/pytorch/pytorch/issues/101850
    os.sched_setaffinity(0, range(os.cpu_count()))

    # Check parameters and modify e.g. metadata
    parameters = check_parameters(parameters=parameters)

    # Load model
    model = sbm.VariableLengthPhaseNet(phases="PSN",
                                       in_samples=parameters["nsamples"],
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
                                                      model=model)

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        device = subgroup_comm.rank % GPUS_PER_NODE
    else:
        device = "cpu"

    # Auf Juwels muss als device "cuda" benutzt werden
    model = model.to("cuda")
    # Au Haicore werden alle gesehen, deswegen wird device gewaehlt
    # model = model.to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)  # Wrap model with DDP.

    # Start training
    # specify loss function
    loss_fn = VectorCrossEntropyLoss()

    # specify learning rate and optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=parameters["learning_rate"])


    train_loss, val_loss = train_model_propulate(model=model,
                                                 patience=parameters.get("patience"),
                                                 epochs=parameters["epochs"],
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 train_loader=train_loader,
                                                 validation_loader=val_loader,
                                                 lr_scheduler=None,
                                                 trace_func=log.info)

    # Return best validation loss as an individual's loss (trained so lower is better).
    dist.destroy_process_group()

    # If parameters for model do not fit and neither training or validation was possible
    # These case are catched by trial and error statement
    if len(val_loss) == 0:
        return 1000
    else:
        min_loss = min(val_loss)

    if is_nan(min_loss):
        return 1000
    return min_loss