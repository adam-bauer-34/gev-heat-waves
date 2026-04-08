import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from config import DATA_ROOT
from src.utils import compute_ecdf
from src.plotting_presets import get_presets
from ambpy.plotutils import make_figure_filename

presets, _ = get_presets(markers=False)
plt.rcParams.update(presets)
np.random.seed(4)

save_figs = True


def plot_kuiper_max_min_ecdf(ds_max, ds_boot, k_type, k_max=0.45, xlim=False,
                             filename_args=None, save_fig=True):
    obs_k_max = ds_max['obs_k_' + k_type].values.flatten()
    syn_k_max = ds_max['syn_k_' + k_type].values.flatten()

    # ignore -1 values (ocean), and restrict to upper bound if requested
    obs_k_max = obs_k_max[(obs_k_max >= 0.0) & (obs_k_max < k_max)]
    syn_k_max = syn_k_max[(syn_k_max >= 0.0) & (syn_k_max < k_max)]

    boot_k = ds_boot['boot_ks'].values.flatten()

    obs_k_max_cdf, obs_k_max_probs = compute_ecdf(obs_k_max, extend_lower=True, extend_upper=False)
    syn_k_max_cdf, syn_k_max_probs = compute_ecdf(syn_k_max, extend_lower=True, extend_upper=False)
    boot_k_cdf, boot_k_probs = compute_ecdf(
        boot_k,
        extend_lower=True,
        extend_upper=True,
        ub=max(max(syn_k_max_cdf), max(obs_k_max_cdf)),
    )

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)

    # first panel: maximum temperatures
    ax.plot(obs_k_max_cdf, obs_k_max_probs, label='Fitted GEV vs.\nEmpirical ERA5',
            linewidth=3)
    ax.plot(syn_k_max_cdf, syn_k_max_probs, label='Fitted GEV vs.\nBootstrapped Samples',
            linewidth=2, linestyle='dashed')
    ax.plot(boot_k_cdf, boot_k_probs, label='Specified GEV vs.\nBootstrapped Samples',
            linewidth=2, linestyle='dotted')

    ax.set_xlabel('Kuiper statistic')
    ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    if xlim:
        ax.set_xlim((0, max(
            max(syn_k_max_cdf),
            max(obs_k_max_cdf),
            max(boot_k_cdf),
        )))

    trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
    ax.text(0.05, 0.97, 'A', transform=ax.transAxes + trans, fontsize=20, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))

    if save_fig and filename_args is not None:
        figpath = make_figure_filename(*filename_args)
        fig.savefig(figpath, dpi=300)
        print(f'Figure saved to: {figpath}')

    plt.close(fig)


def load_datasets(tmin, anom_type):

    ds_max_path = DATA_ROOT / 'ERA5' / 'gev' / f'era5_t2m_annual_max_1deg_landonly_gev_stat_TMIN{tmin}_{anom_type}_kuiper.nc'
    ds_boot_path = DATA_ROOT / 'stats' / f'bootstrapped_ks_{tmin}.nc'

    print(f'Loading ds_max: {ds_max_path}')
    print(f'Loading ds_boot: {ds_boot_path}')

    ds_max = xr.open_dataset(ds_max_path, engine='netcdf4')
    ds_boot = xr.open_dataset(ds_boot_path, engine='netcdf4')

    return ds_max, ds_boot


def main():
    anom_types = ['annmean', 'trend', 'raw']
    tmins = [1950, 1979]
    k_max = 0.45

    for tmin in tmins:
        for anom in anom_types:
            ds_max, ds_boot = load_datasets(tmin, anom)

            plot_tag = "raw" if anom == 'raw' else anom
            filename_args = [f'kuiper_max_{plot_tag}_tmin{tmin}', 'png', 'figs/analysis']

            plot_kuiper_max_min_ecdf(
                ds_max,
                ds_boot,
                anom if anom == 'raw' else f'anom_{anom}',
                k_max=k_max,
                xlim=True,
                filename_args=filename_args,
                save_fig=save_figs,
            )

            ds_max.close()
            ds_boot.close()


if __name__ == '__main__':
    main()
