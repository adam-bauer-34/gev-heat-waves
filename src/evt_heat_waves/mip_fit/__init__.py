from .most.most import runner as most_runner
from .prim.prim import runner as prim_runner

FIT_REGISTRY = {
    'most': {
        'no_mpi': most_runner,
        'mpi': None
    },
    'prim': {
        'no_mpi': prim_runner,
        'mpi': None
    }
}