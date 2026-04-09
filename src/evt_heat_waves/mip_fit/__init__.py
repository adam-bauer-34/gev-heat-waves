from .most.most import runner as most_runner
from .most.most_mpi import runner as most_mpi_runner

from .prim.prim import runner as prim_runner
from .prim.prim_mpi import runner as prim_mpi_runner

FIT_REGISTRY = {
    'most': {
        'no_mpi': most_runner,
        'mpi': most_mpi_runner
    },
    'prim': {
        'no_mpi': prim_runner,
        'mpi': prim_mpi_runner
    }
}