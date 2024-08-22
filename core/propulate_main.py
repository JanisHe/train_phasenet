import logging
import pathlib
import random

from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from utils import ind_loss


GPUS_PER_NODE: int = 4  # This example script was tested on a single node with 4 GPUs.
SUBGROUP_COMM_METHOD = "nccl-slurm"
log_path = "torch_ckpts"
log = logging.getLogger("propulate")  # Get logger instance.


if __name__ == "__main__":
    generations = 10
    num_islands = 2  # Number of islands
    migration_probability = 0.9  # Migration probability
    pollination = True  # Whether to use pollination or migration
    checkpoint_path = "./propulate_ckpt"

    comm = MPI.COMM_WORLD

    pop_size = 2 * comm.size  # Breeding population size
    limits_dict = {"learning_rate": (0.0001, 0.01),
                   "batch_size": (512, 1024)}
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
        log.info("Starting Torch DDP tutorial!")

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
        generations=generations,  # Overall number of generations
        num_islands=num_islands,  # Number of islands
        migration_probability=migration_probability,  # Migration probability
        pollination=pollination,  # Whether to use pollination or migration
        checkpoint_path=checkpoint_path,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS -----
        ranks_per_worker=GPUS_PER_NODE,  # Number of ranks per (multi rank) worker
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

