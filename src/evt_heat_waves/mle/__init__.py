"""Last edited: 4/30/2026, 7:36 PM CST
"""

from .hess import get_hessian_stat
from .hess import get_hessian_nonstat
from .hess import get_hessian_stat_fix_shape

# define registry of valid hessians to use to calculate standard errors of MLE
HESS_REGISTRY = {
    'stat': get_hessian_stat,
    'stat_new': get_hessian_stat,
    'nonstat': get_hessian_nonstat,
    'stat_fix_shape': get_hessian_stat_fix_shape
}