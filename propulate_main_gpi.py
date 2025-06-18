import os
import sys
import pathlib
import shutil
import random

import seisbench  # noqa
import logging
import yaml

import seisbench.data as sbd # noqa
import seisbench.generate as sbg # noqa
import seisbench.models as sbm # noqa
from seisbench.util import worker_seeding # noqa
from seisbench.models.phasenet import PhaseNet # noqa
from mpi4py import MPI
from propulate import Propulator
from propulate.utils import get_default_propagator, set_logger_config

from core.utils import check_parameters, check_propulate_limits
from core.propulate_gpi import ind_loss


def main(parfile: str):
    comm = MPI.COMM_WORLD
    comm.Barrier()

    # Read parameters from yaml file
    with open(parfile, "r") as f:
        params = yaml.safe_load(f)

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
                   "loss_fn": params["loss_fn"],
                   "parfile": renamed_parfile}

    # Check whether one parameter in limits_dict has only a length of one
    # If yes, the same value is appended to the tuple
    limits_dict = check_propulate_limits(params=limits_dict)
    rng = random.Random(comm.rank)  # Set up separate random number generator for evolutionary optimizer.

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file=f"logs/{pathlib.Path(parfile).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits_dict,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=ind_loss,  # Loss function to optimize
        propagator=propagator,  # Evolutionary operator
        rng=rng,  # Random number generator
        island_comm=comm,  # Communicator
        generations=params["generations"],  # Number of generations per worker
        checkpoint_path=params["checkpoint_path"],  # Path to save checkpoints to
    )

    # Run optimization and print summary of results.
    propulator.propulate(
        logging_interval=1,
        debug=2,  # Logging interval and verbosity level
    )
    propulator.summarize(
        top_n=5,
        debug=2,  # Print top-n best individuals on each island in summary.
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