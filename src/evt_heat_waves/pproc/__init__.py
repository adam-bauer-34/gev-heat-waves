from .pproc_era5 import pproc_era5
from .pproc_cmip import pproc_cmip
from .pproc_amip import pproc_amip

PPROC_REGISTRY = {
    'era5': {
        'runner': pproc_era5,
    },
    'cmip': {
        'runner': pproc_cmip,
    },
    'amip': {
        'runner': pproc_amip,
    }
}