from .pproc_era5.runner import pproc_era5
from .pproc_cmip.runner import pproc_cmip
from .pproc_amip.runner import pproc_amip

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