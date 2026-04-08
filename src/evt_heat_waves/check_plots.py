"""Check plot codes.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# import custom plotting stuff
from ambpy.plotutils import make_figure_filename
# plt.style.use('ambpy')

def plot_side_by_side(data, data2, titles=("Dataset 1", "Dataset 2"),
                      save_figs=False,
                      filename_args=['check_plot', 'png', 'figs']):
    """
    Plot two xarray DataArrays side by side on world maps using Cartopy.

    Parameters
    ----------
    data : xr.DataArray
        First variable to plot (e.g., land-sea mask)
    data2 : xr.DataArray
        Second variable to plot
    titles : tuple of str
        Titles for the two subplots
    save_figs : bool
        If True, saves the figure instead of just showing it
    """

    # --- Create the figure and subplots ---
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(14, 5),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for ax, da, title in zip(axes, [data, data2], titles):
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        # Plot the data
        im = da.plot(
            ax=ax,
            transform=ccrs.PlateCarree())

        ax.set_title(title)

    # Add a shared colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    # cbar.set_label(val_plotted)

    # Layout and save/show
    # plt.tight_layout()
    if save_figs:
        plt.savefig(make_figure_filename(*filename_args), dpi=300)

    plt.close()  # close to avoid memory issues