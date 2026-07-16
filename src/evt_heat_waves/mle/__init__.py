"""Last edited: 4/30/2026, 7:36 PM CST"""

from .hess import get_hessian_stat
from .hess import get_hessian_nonstat
from .hess import get_hessian_stat_fix_shape
from .hess import get_hessian_nonstat_only_loc_trend
from .hess import get_hessian_nonstat_only_loc_trend_fix_shape
from .hess import get_hessian_nonstat_only_loc_and_shape_trend_fix_shape

# define registry of valid hessians to use to calculate standard errors of MLE
HESS_REGISTRY = {
    "stat": get_hessian_stat,
    "stat_new": get_hessian_stat,
    "stat_lax": get_hessian_stat,
    "nonstat": get_hessian_nonstat,
    "stat_gumbel": get_hessian_stat_fix_shape,
    "stat_minxi": get_hessian_stat_fix_shape,
    "nonstat_gumbel_only_loc_trend": get_hessian_nonstat_only_loc_trend_fix_shape,
    "nonstat_minxi_only_loc_trend": get_hessian_nonstat_only_loc_trend_fix_shape,
    "nonstat_kleinxi_only_loc_trend": get_hessian_nonstat_only_loc_trend_fix_shape,
    "nonstat_only_loc_trend": get_hessian_nonstat_only_loc_trend,
    "nonstat_gumbel_only_loc_and_shape_trend": get_hessian_nonstat_only_loc_and_shape_trend_fix_shape,
}
