import sys
import os
import shutil
import logging
import pathlib
import random

import yaml
from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from core.propulate_functions import ind_loss
from core.utils import check_propulate_limits, check_parameters


GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
SUBGROUP_COMM_METHOD = "nccl-slurm"
log = logging.getLogger("propulate")  # Get logger instance.


def main(parfile: str):
    """

    """
    # Read parameters from yaml file
    with open(parfile, "r") as f:
        params = yaml.safe_load(f)

    comm = MPI.COMM_WORLD

    pop_size = 2 * comm.size  # Breeding population size

    # Check params and if key is not found, use default value
    params = check_parameters(parameters=params)

    # Create a copy of parfile and change str for parfile
    # Otherwise, if someone makes changes in parfile, these changes are read by ind_loss (everytime!)
    renamed_parfile = os.path.join(params["checkpoint_path"], os.path.split(parfile)[-1])
    if not os.path.exists(params["checkpoint_path"]):
        try:
            os.makedirs(params["checkpoint_path"])
        except FileExistsError:
            pass
    try:
        shutil.copyfile(src=parfile,
                        dst=renamed_parfile)
    except shutil.SameFileError:
        pass

    # Create file to store parameters for failed models and successfull model
    f_tested = open(os.path.join(params["checkpoint_path"], "tested_models"), "w")
    f_tested.close()
    f_failed = open(os.path.join(params["checkpoint_path"], "failed_models"), "w")
    f_failed.close()

    # TODO: Write h_params to yaml
    limits_dict = {"learning_rate": params["learning_rate"],
                   "batch_size": params["batch_size"],
                   "nsamples": params["nsamples"],
                   "kernel_size": params["kernel_size"],
                   "filter_factor": params["filter_factor"],
                   "depth": params["depth"],
                   "drop_rate": params["drop_rate"],
                   "stride": params["stride"],
                   "filters_root": params["filters_root"],
                   "activation_function": params["activation_function"],
                   "parfile": renamed_parfile}

    # Check whether one parameter in limits_dict has only a length of one
    # If yes, the same value is appended to the tuple
    limits_dict = check_propulate_limits(params=limits_dict)

    rng = random.Random(
        comm.rank
    )  # Set up separate random number generator for evolutionary optimizer.

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"logs/{pathlib.Path(parfile).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )
    if comm.rank == 0:
        log.info("Starting DDP + Propulate on PhaseNet!")

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits_dict,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up island model.
    islands = Islands(
        loss_fn=ind_loss,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=params["generations"],  # Overall number of generations
        num_islands=params["num_islands"],  # Number of islands
        migration_probability=params["migration_probability"],  # Migration probability
        pollination=params["pollination"] ,  # Whether to use pollination or migration
        checkpoint_path=params["checkpoint_path"],  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS -----
        ranks_per_worker=params["ranks_per_worker"]  # GPUS_PER_NODE,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.propulate(
        logging_interval=1,  # Logging interval
        debug=1,  # Debug level
    )
    islands.summarize(
        top_n=5,  # Print top-n best individuals on each island in summary.
        debug=1,  # Debug level
    )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        parfile = "./propulate_parfile.yml"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Run main function
    main(parfile=parfile)
