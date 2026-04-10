from .fit.fit_mpi import runner as mpi_runner
from .fit.fit import runner as no_mpi_runner

FIT_REGISTRY = {
    'no_mpi': no_mpi_runner,
    'mpi': mpi_runner
}