import numpy as np
import matplotlib.pyplot as plt 

from evt_heat_waves.config import MLE_FIT_ATTRS, CHECKS_PATH, FIGS_PATH
from evt_heat_waves.plotting.utils import make_figure_filename

# Set color and marker sets for CMIP models
# 10 High-Contrast, Colorblind-Friendly Hex Codes (Paul Tol / Okabe-Ito)
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', 
          '#AA3377', '#EE7733', '#009988', '#332288', '#BBBBBB']

# Marker Set
markers = ['o', 's', 'D'] # Circle, Square, Diamond

title_map = {
    'loc': 'Location Parameter',
    'loc_t': 'Location Parameter Trend',
    'scale': 'Scale Parameter',
    'scale_t': 'Scale Parameter Trend',
    'shape': 'Shape Parameter',
    'shape_t': 'Shape Parameter Trend'
}

xlabel_map = {
    'loc': r'$\mu^{ERA5}_0$ ($^\circ$C)',
    'loc_t': r"$\mu^{ERA5}_1'$ ($^\circ$C/decade)",
    'scale': r'$\sigma^{ERA5}_0$ ($^\circ$C)',
    'scale_t': r"$\sigma^{ERA5}_1'$ ($^\circ$C/decade)",
    'shape': r'$\xi^{ERA5}_0$ ($-$)',
    'shape_t': r"$\xi^{ERA5}_1'$ (decade$^{-1}$)"
}

ylabel_maps = {
    'cmip': {
        'loc': r'$\mu^{CMIP}_0$ ($^\circ$C)',
        'loc_t': r"$\mu^{CMIP}_1'$ ($^\circ$C/decade)",
        'scale': r'$\sigma^{CMIP}_0$ ($^\circ$C)',
        'scale_t': r"$\sigma^{CMIP}_1'$ ($^\circ$C/decade)",
        'shape': r'$\xi^{CMIP}_0$ ($-$)',
        'shape_t': r"$\xi^{CMIP}_1'$ (decade$^{-1}$)"
    },
    'amip': {
        'loc': r'$\mu^{AMIP}_0$ ($^\circ$C)',
        'loc_t': r"$\mu^{AMIP}_1'$ ($^\circ$C/decade)",
        'scale': r'$\sigma^{AMIP}_0$ ($^\circ$C)',
        'scale_t': r"$\sigma^{AMIP}_1'$ ($^\circ$C/decade)",
        'shape': r'$\xi^{AMIP}_0$ ($-$)',
        'shape_t': r"$\xi^{AMIP}_1'$ (decade$^{-1}$)"   
    }
}

corr_title_map = {
    'loc': r'Location Parameter | $\mu_0$ $(^\circ$C$)$',
    'loc_t': r"Location Parameter Trend | $\mu_1$ $(^\circ$C / dec$)$",
    'scale': r'Scale Parameter | $\sigma_0$ $(^\circ$C$)$',
    'scale_t': r"Scale Parameter Trend | $\sigma_1$ $(^\circ$C / dec$)$",
    'shape': r'Shape Parameter | $\xi_0$ $(-)$',
    'shape_t': r"Shape Parameter Trend | $\xi_1$ $($dec$^{-1})$"
}

panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# mapping for number of parameters to grid shape and figure size
param_num_to_grid = {
    2: {'grid': (1, 2), 'figsize': (12, 6)},
    3: {'grid': (1, 3), 'figsize': (18, 6)},
    4: {'grid': (2, 2), 'figsize': (12, 12)},
    5: {'grid': (2, 3), 'figsize': (18, 12)}
}

xlabels = {
    'med': "Median absolute deviation: MODEL $-$ ERA5",
    'mean': "Mean absolute deviation: MODEL $-$ ERA5"
}

corr_plot_attrs = {
    2: {
        'grid': (1, 2),
        'figsize': (14, 6),
        'have_ylabels': [0],
        'have_xlabels': [0, 1],
        'legend_panel': 1,
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (0.5, 1.15),
            'ncol': 5,
        },
    },
    3: {
        'grid': (1, 3),
        'figsize': (25, 7),
        'have_ylabels': [0],
        'have_xlabels': [0, 1, 2],
        'legend_panel': 1,
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (0.5, -0.36),
            'ncol': 5,
            'frameon': True
        },
    },
    4: {
        'have_ylabels': [0, 2],
        'have_xlabels': [2, 3],
        'legend_panel': 3,
        'legend': {
            'loc': 'center',
            'bbox_to_anchor': (0.5, 1.15),
            'ncol': 5,
        },
    }
}

def plot_bias_vs_corr(abs_dev_prim, r2s_prim, 
                      abs_dev_most, r2s_most,
                      models, model_with_most, fit, med_or_mean,
                      fname: str, save_figs: bool = True):
    fig, ax = plt.subplots(
        *corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['grid'],
        figsize=corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['figsize']
    )

    for (idx, (a, var, param)) in enumerate(zip(ax.flatten(), r2s_prim.keys(), MLE_FIT_ATTRS[fit]['param_names'])):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        for mdx, m in enumerate(models):
            marker = markers[mdx // 10]  # choose marker
            color = colors[mdx % 10]  # choose color

            a.scatter(abs_dev_prim[var][mdx], r2s_prim[var][mdx], s=90, marker=marker, color=color, label=m.name, zorder=100)
            a.set_title(corr_title_map[param])
            if idx in corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['have_ylabels']:
                a.set_ylabel("$r^2$")
            
            if idx in corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['have_xlabels']:
                a.set_xlabel(xlabels[med_or_mean])


    for (idx, (a, var, param)) in enumerate(zip(ax.flatten(), r2s_most.keys(), MLE_FIT_ATTRS[fit]['param_names'])):
        a.axvline(0, 0, 1, linestyle='solid', color='k')
        a.axhline(0, -1, 1, linestyle='solid', color='k')
        for mdx in range(len(abs_dev_most[var])):
            a.scatter(abs_dev_most[var][mdx], r2s_most[var][mdx],
                      s=60, marker='.', color='grey', zorder=1,
                      label=f'{model_with_most} Ensemble Members' if mdx == 0 else None)
            a.set_title(corr_title_map[param])
            if idx in corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['have_ylabels']:
                a.set_ylabel("r$^2$")
            
            if idx in corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['have_xlabels']:
                a.set_xlabel(xlabels[med_or_mean])

    ax[corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['legend_panel']].legend(
        **corr_plot_attrs[MLE_FIT_ATTRS[fit]['N_params']]['legend']
    )

    for a, label in zip(ax.flatten(), panel_labels):
        a.text(0.025, 0.97, label, transform=a.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')

    if save_figs:
        fname = make_figure_filename(fname, outdir=FIGS_PATH)
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fname}")

def plot_scatter_regression(x_data, y_data, slopes, intercepts, r2s,
                            fit, mip, model_name,
                            fname: str):
    ylabel_map = ylabel_maps[mip]  # map for y labels

    # Set up the plotting grid
    fig, ax = plt.subplots(
        *param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['grid'],
        figsize=param_num_to_grid[MLE_FIT_ATTRS[fit]['N_params']]['figsize']
    )

    for (a, param, x, y, slope, intercept, r2) in zip(ax.flatten(), MLE_FIT_ATTRS[fit]['param_names'], x_data, y_data, slopes, intercepts, r2s):
        # data and regression lines
        a.scatter(x, y, s=1, marker='.', c='grey')  # data
        a.plot(np.arange(min(x), max(x), (max(x) - min(x))/1000),
                np.arange(min(x), max(x), (max(x) - min(x))/1000),
                linewidth=2.5, linestyle='dashed', color='r')  # one to one line
        a.plot(x, slope * x + intercept, linestyle='solid', color='b', linewidth=2.5, label=f"r$^2$={r2:.2f}")  # regression line

        # aesthetics
        a.set_title(title_map[param])
        a.set_xlabel(xlabel_map[param])
        a.set_ylabel(ylabel_map[param])
        a.legend()
    
    fig.suptitle(f"Model: {model_name}")
    fig.tight_layout()

    fname = make_figure_filename(fname, outdir=CHECKS_PATH)
    fig.savefig(fname, dpi=300)
    print(f"     Figure saved to: {fname}")

def mutual_mask_perc(x, y, p_lo, p_hi):
    xnew = x[
        (x >= np.nanpercentile(x, p_lo)) & (x <= np.nanpercentile(x, p_hi))
    ]
    ynew = y[
        (x >= np.nanpercentile(x, p_lo)) & (x <= np.nanpercentile(x, p_hi))
    ]

    yfinal = ynew[
        (ynew >= np.nanpercentile(y, p_lo)) & (ynew <= np.nanpercentile(y, p_hi))
    ]

    xfinal = xnew[
        (ynew >= np.nanpercentile(y, p_lo)) & (ynew <= np.nanpercentile(y, p_hi))
    ]

    return xfinal, yfinal