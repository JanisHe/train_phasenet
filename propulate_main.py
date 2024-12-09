import sys
import os
import logging
import pathlib
import random

import yaml
from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from core.torch_functions import ind_loss


GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
SUBGROUP_COMM_METHOD = "nccl-slurm"
log_path = "torch_ckpts"
# TODO: Write output into file to save best parameters
log = logging.getLogger("propulate")  # Get logger instance.


def main(parfile: str):
    """

    """
    # Read parameters from yaml file
    with open(parfile, "r") as f:
        params = yaml.safe_load(f)

    comm = MPI.COMM_WORLD

    pop_size = 2 * comm.size  # Breeding population size
    # TODO: Write h_params to yaml
    limits_dict = {"learning_rate": tuple(params["learning_rate"]),
                   "batch_size": tuple(params["batch_size"]),
                   "nsamples": tuple(params["nsamples"]),
                   "kernel_size": tuple(params["kernel_size"]),
                   "depth": tuple(params["depth"]),
                   "drop_rate": tuple(params["drop_rate"]),
                   "stride": tuple(params["stride"]),
                   "filters_root": tuple(params["filters_root"]),
                   "activation_function": tuple(params["activation_function"]),
                   "parfile": (parfile, parfile)}

    rng = random.Random(
        comm.rank
    )  # Set up separate random number generator for evolutionary optimizer.

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"{log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
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
        ranks_per_worker=2  # GPUS_PER_NODE,  # Number of ranks per (multi rank) worker
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
