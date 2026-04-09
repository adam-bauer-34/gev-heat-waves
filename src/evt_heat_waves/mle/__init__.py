from .hess import get_hessian_stat
from .hess import get_hessian_nonstat

# define registry of valid hessians to use to calculate standard errors of MLE
HESS_REGISTRY = {
    'stat': get_hessian_stat,
    'nonstat': get_hessian_nonstat
}