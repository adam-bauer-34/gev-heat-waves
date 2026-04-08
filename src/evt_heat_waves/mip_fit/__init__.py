from .cmip.most.most import runner as cmip_most_runner
from .cmip.most.most_mpi import runner as cmip_most_mpi_runner

from .cmip.prim.prim import runner as cmip_prim_runner
from .cmip.prim.prim_mpi import runner as cmip_prim_mpi_runner

from .amip.most.most import runner as amip_most_runner
from .amip.most.most_mpi import runner as amip_most_mpi_runner

from .amip.prim.prim import runner as amip_prim_runner
from .amip.prim.prim_mpi import runner as amip_prim_mpi_runner


FIT_REGISTRY = {
    'cmip': {
        'most': {
            'no_mpi': cmip_most_runner,
            'mpi': cmip_most_mpi_runner
        },
        'prim': {
            'no_mpi': cmip_prim_runner,
            'mpi': cmip_prim_mpi_runner
        },
    },
    'amip': {
        'most': {
            'no_mpi': amip_most_runner,
            'mpi': amip_most_mpi_runner
        },
        'prim': {
            'no_mpi': amip_prim_runner,
            'mpi': amip_prim_mpi_runner
        },
    }
}