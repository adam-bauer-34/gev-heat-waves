from .fit.fit import runner as no_mpi_runner

from .kuiper.kuipers import runner as no_mpi_runner

FIT_REGISTRY = {
    'no_mpi': no_mpi_runner
}

KUIPER_REGISTRY = {
    'no_mpi': no_mpi_runner
}