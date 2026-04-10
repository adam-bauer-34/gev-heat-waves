from .fit.fit_mpi import runner as mpi_runner
from .fit.fit import runner as no_mpi_runner

from .kuiper.kuipers_mpi import runner as mpi_runner_k
from .kuiper.kuipers import runner as no_mpi_runner_k

FIT_REGISTRY = {
    'no_mpi': no_mpi_runner,
    'mpi': mpi_runner
}

KUIPER_REGISTRY = {
    'no_mpi': no_mpi_runner_k,
    'mpi': mpi_runner_k
}