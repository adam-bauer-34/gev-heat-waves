from .fit_kuiper_mpi import runner as mpi_runner
from .fit_kuiper import runner as no_mpi_runner

FIT_REGISTRY = {
    'no_mpi': no_mpi_runner,
    'mpi': mpi_runner
}