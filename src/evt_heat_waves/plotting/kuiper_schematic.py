"""Two-panel Kuiper statistic figure: schematic of the calculation (A) next to
the ECDFs of the observed and misspecification Kuiper statistics (B).

Panel A is a flowchart of what actually happens in
``evt_heat_waves.era5.kuiper.kuiper_fitting``: one branch produces ``obs_k``,
the other produces ``mis_k`` via the common-random-number draw from the
free-shape fit. Panel B is the figure previously made by
``plot_kuiper_max_min_ecdf_alt`` in the analysis notebook.

Adam Michael Bauer
UChicago
"""

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from evt_heat_waves.utils import compute_ecdf
from evt_heat_waves.plotting.utils import make_figure_filename

# ---------------------------------------------------------------------------
# style constants: kept together so the schematic and the ECDF panel can be
# recolored in one place
# ---------------------------------------------------------------------------

# colorblind-safe palette; matches the first two entries of the prop_cycle in
# plotting_presets so the schematic branches read as the panel B curves
OBS_COLOR = "#000000"
MIS_COLOR = "#E69F00"
SHARED_COLOR = "#4D4D4D"

# box fills: very light tints of the branch colors
OBS_FILL = "#EDEDED"
MIS_FILL = "#FDF1DC"
SHARED_FILL = "#FFFFFF"

BOX_LW = 1.8
ARROW_LW = 1.8
BOX_FONTSIZE = 12.5
RESULT_FONTSIZE = 16
PANEL_LABEL_FONTSIZE = 18

# the boxstyle pad is drawn *outside* the nominal box rect, so the layout below
# has to leave room for it or the top box gets clipped at the axes edge
BOX_PAD = 0.006
BOXSTYLE = f"round,pad={BOX_PAD},rounding_size=0.012"

# ---------------------------------------------------------------------------
# schematic geometry, in axes-fraction coordinates of panel A
# ---------------------------------------------------------------------------

# the two branches sit in the left/right columns; the shared input spans both
COL_L, COL_R = 0.255, 0.745
BOX_W = 0.45
RESULT_W = 0.30

# the shared input box spans both columns exactly, so their outer edges line up
INPUT_W = (COL_R - COL_L) + BOX_W

# row heights: two- vs three-line boxes, plus the terminal result boxes
H_SMALL = 0.095
H_LARGE = 0.130
H_RESULT = 0.085

# vertical stacking. The right branch is the long one, so it sets the rows; the
# left branch reuses the rows it needs and runs a long arrow to the result.
STACK_TOP = 1.0 - BOX_PAD
ROW_GAP = 0.062
STACK_HEIGHTS = [H_SMALL, H_SMALL, H_LARGE, H_SMALL, H_LARGE, H_RESULT]


def _stack_centers(top, heights, gap):
    """Vertical box centers for a stack running downward from ``top``."""

    centers, y = [], top
    for h in heights:
        centers.append(y - h / 2)
        y -= h + gap

    return centers


# row centers: [input, fit, draw, refit, kuiper, result]
ROWS = _stack_centers(STACK_TOP, STACK_HEIGHTS, ROW_GAP)
ROW_INPUT, ROW_FIT, ROW_DRAW, ROW_REFIT, ROW_KUIPER, ROW_RESULT = ROWS

# the branch elbow sits halfway down the gap below the shared input box
SPLIT_Y = ROW_INPUT - H_SMALL / 2 - ROW_GAP / 2


def _draw_box(
    ax,
    xc,
    yc,
    text,
    facecolor,
    edgecolor,
    width=BOX_W,
    height=H_SMALL,
    fontsize=BOX_FONTSIZE,
    fontweight="normal",
    boxstyle=BOXSTYLE,
    lw=BOX_LW,
    textcolor="black",
    zorder=3,
):
    """Draw a single rounded flowchart box and return its (xc, yc, w, h).

    Coordinates are box centers in axes fraction, which keeps the row/column
    layout above readable as a table of positions.
    """

    ax.add_patch(
        FancyBboxPatch(
            (xc - width / 2, yc - height / 2),
            width,
            height,
            boxstyle=boxstyle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            mutation_aspect=1.0,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )

    ax.text(
        xc,
        yc,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        linespacing=1.45,
        transform=ax.transAxes,
        zorder=zorder + 1,
    )

    return (xc, yc, width, height)


def _arrow(ax, start, end, color, lw=ARROW_LW, mutation_scale=16, linestyle="-"):
    """Draw a straight arrow between two axes-fraction points."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            transform=ax.transAxes,
            zorder=2,
        )
    )


def _draw_result_box(ax, xc, label, color, width=RESULT_W, height=H_RESULT):
    """Draw a terminal (result) box: heavier border, branch-colored label."""

    return _draw_box(
        ax,
        xc,
        ROW_RESULT,
        label,
        "white",
        color,
        width=width,
        height=height,
        fontsize=RESULT_FONTSIZE,
        fontweight="bold",
        lw=BOX_LW + 1.2,
        textcolor=color,
    )


def _arrow_between(ax, box_from, box_to, color, **kwargs):
    """Vertical arrow from the bottom edge of one box to the top of another."""

    x_from, y_from, _, h_from = box_from
    x_to, y_to, _, h_to = box_to

    _arrow(
        ax,
        (x_from, y_from - h_from / 2),
        (x_to, y_to + h_to / 2),
        color,
        **kwargs,
    )


def draw_kuiper_schematic(ax, n_reps_label=None):
    """Draw the flowchart explaining how obs_k and mis_k are computed.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axes to draw into; all spines/ticks are removed

    n_reps_label: str or None
        replicate count to name in the averaging line of the final
        misspecification box. None (the n_reps=1 default of the pipeline) drops
        that line, since there's nothing to average over
    """

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ---- shared input -----------------------------------------------------
    box_data = _draw_box(
        ax,
        0.5,
        ROW_INPUT,
        "ERA5 annual maximum $T_{2m}$ anomalies\n"
        r"at one land gridcell: "
        r"$\{x_t\}_{t=T_{\mathrm{min}}}^{N_{\mathrm{years}}}$",
        SHARED_FILL,
        SHARED_COLOR,
        width=INPUT_W,
    )

    # split the shared input into the two branches with an elbow. linestyle is
    # forced: the rcParams prop_cycle otherwise hands these lines 'dashed'
    ax.plot(
        [0.5, 0.5],
        [box_data[1] - H_SMALL / 2, SPLIT_Y],
        color=SHARED_COLOR,
        lw=ARROW_LW,
        linestyle="solid",
        transform=ax.transAxes,
        zorder=2,
        solid_capstyle="butt",
    )
    ax.plot(
        [COL_L, COL_R],
        [SPLIT_Y, SPLIT_Y],
        color=SHARED_COLOR,
        lw=ARROW_LW,
        linestyle="solid",
        transform=ax.transAxes,
        zorder=2,
        solid_capstyle="round",
    )

    # ---- left branch: observed Kuiper statistic ---------------------------
    box_fit_fix = _draw_box(
        ax,
        COL_L,
        ROW_FIT,
        r"Fit Gumbel to $\{x_t\}$ using MLE" "\n"
        r"($\xi \equiv 0$): $\hat{\mu},\ \hat{\sigma}$",
        OBS_FILL,
        OBS_COLOR,
    )
    _arrow(ax, (COL_L, SPLIT_Y), (COL_L, ROW_FIT + H_SMALL / 2), OBS_COLOR)

    box_k_obs = _draw_box(
        ax,
        COL_L,
        ROW_DRAW,
        "Kuiper distance between\n"
        r"empirical CDF of $\{x_t\}$ and"
        "\n"
        r"Gumbel($\hat{\mu},\hat{\sigma}$) CDF",
        OBS_FILL,
        OBS_COLOR,
        height=H_LARGE,
    )
    _arrow_between(ax, box_fit_fix, box_k_obs, OBS_COLOR)

    # ---- right branch: misspecification Kuiper statistic ------------------
    box_fit_free = _draw_box(
        ax,
        COL_R,
        ROW_FIT,
        r"Fit GEV to $\{x_t\}$ with $\xi$ free using MLE" "\n"
        r"$\tilde{\xi},\ \tilde{\mu},\ \tilde{\sigma}$",
        MIS_FILL,
        MIS_COLOR,
    )
    _arrow(ax, (COL_R, SPLIT_Y), (COL_R, ROW_FIT + H_SMALL / 2), MIS_COLOR)

    box_draw = _draw_box(
        ax,
        COL_R,
        ROW_DRAW,
        "Draw $N_{\\mathrm{years}}$ synthetic maxima\n"
        r"$x_t^{*} \sim \mathrm{GEV}(\tilde{\xi},\tilde{\mu},\tilde{\sigma})$",
        MIS_FILL,
        MIS_COLOR,
        height=H_LARGE,
    )
    _arrow_between(ax, box_fit_free, box_draw, MIS_COLOR)

    box_refit = _draw_box(
        ax,
        COL_R,
        ROW_REFIT,
        r"Fit Gumbel to $\{x_t^{*}\}$ using MLE"
        "\n"
        r"($\xi \equiv 0$): $\hat{\mu}^{*},\ \hat{\sigma}^{*}$",
        MIS_FILL,
        MIS_COLOR,
    )
    _arrow_between(ax, box_draw, box_refit, MIS_COLOR)

    k_mis_text = (
        "Kuiper distance between\n"
        r"empirical CDF of $\{x_t^{*}\}$ and"
        "\n"
        r"Gumbel($\hat{\mu}^{*},\hat{\sigma}^{*}$) CDF"
    )
    if n_reps_label is not None:
        k_mis_text += f",\naveraged over {n_reps_label} replicates"

    box_k_mis = _draw_box(
        ax,
        COL_R,
        ROW_KUIPER,
        k_mis_text,
        MIS_FILL,
        MIS_COLOR,
        height=H_LARGE,
    )
    _arrow_between(ax, box_refit, box_k_mis, MIS_COLOR)

    # ---- results ----------------------------------------------------------
    box_v_obs = _draw_result_box(ax, COL_L, r"$V_{\mathrm{obs}}$", OBS_COLOR)
    _arrow_between(ax, box_k_obs, box_v_obs, OBS_COLOR)

    box_v_mis = _draw_result_box(ax, COL_R, r"$V_{\mathrm{syn}}$", MIS_COLOR)
    _arrow_between(ax, box_k_mis, box_v_mis, MIS_COLOR)

    # label the long left arrow rather than leaving it bare: the white bbox
    # breaks the shaft, which reads as an edge annotation
    ax.text(
        COL_L,
        (ROW_DRAW + ROW_RESULT) / 2,
        "Departure of the\nobservations from\nthe fitted model",
        ha="center",
        va="center",
        fontsize=11.5,
        style="italic",
        color=SHARED_COLOR,
        linespacing=1.4,
        transform=ax.transAxes,
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=4.0),
    )

    # closing note tying the schematic to panel B
    ax.text(
        0.5,
        ROW_RESULT - H_RESULT / 2 - 0.035,
        "Repeated at every land gridcell "
        r"$\longrightarrow$ Distributions in panel B",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color=SHARED_COLOR,
        transform=ax.transAxes,
    )


def plot_kuiper_schematic_ecdf(
    ds_max,
    k_type,
    k_min=0.09,
    k_max=0.38,
    xlim=True,
    n_reps_label=None,
    figsize=(16.5, 8.0),
    save_figs=False,
    filename_args=("kuiper_schematic", None, "png"),
):
    """Two-panel figure: schematic of the Kuiper calculation + ECDFs.

    Parameters
    ----------
    ds_max: xarray.Dataset
        the Kuiper output dataset, containing 'obs_k_{k_type}' and
        'mis_k_{k_type}'

    k_type: str
        the anomaly-type suffix (e.g., 'anom_annmean')

    k_min, k_max: float
        x-axis limits for panel B; k_max also caps the values kept, matching
        the notebook version of this plot

    xlim: bool
        whether to apply (k_min, k_max) to panel B

    n_reps_label: str or None
        replicate count named in the schematic; None matches the n_reps=1
        default of the Kuiper pipeline

    figsize: tuple
        figure size in inches

    save_figs: bool
        whether to write the figure to disk

    filename_args: tuple
        passed to make_figure_filename as (name, outdir, ext)

    Returns
    -------
    fig, (ax_schematic, ax_ecdf)
    """

    obs_k = ds_max["obs_k_" + k_type].values.flatten()
    mis_k = ds_max["mis_k_" + k_type].values.flatten()

    # ignore the -1 sentinel (too few years) and the ocean NaNs
    obs_k = obs_k[(obs_k >= 0.0) & (obs_k < k_max)]
    mis_k = mis_k[(mis_k >= 0.0) & (mis_k < k_max)]

    obs_x, obs_p = compute_ecdf(obs_k, extend_lower=True, extend_upper=False)
    mis_x, mis_p = compute_ecdf(mis_k, extend_lower=True, extend_upper=False)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # the extra figure width goes to panel B: panel A's boxes are sized in axes
    # fractions, so narrowing it would crowd the text in the widest boxes
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    # panel B is padded vertically so the ECDF axes doesn't stretch to the full
    # height of the (tall) schematic
    gs_right = gs[0, 1].subgridspec(3, 1, height_ratios=[0.14, 1.0, 0.14])

    ax_a = fig.add_subplot(gs[0, 0])

    # invisible axes over the whole right cell, so the 'B' label sits at the
    # same height as 'A' rather than at the top of the padded ECDF axes
    ax_b_label = fig.add_subplot(gs[0, 1])
    ax_b_label.set_axis_off()

    ax_b = fig.add_subplot(gs_right[1, 0])

    draw_kuiper_schematic(ax_a, n_reps_label=n_reps_label)

    ax_b.plot(
        obs_x,
        obs_p,
        color=OBS_COLOR,
        linestyle="solid",
        linewidth=3,
        label=r"ERA5 versus Gumbel Fit ($V_{\mathrm{obs}}$)",
    )
    ax_b.plot(
        mis_x,
        mis_p,
        color=MIS_COLOR,
        linestyle="dashed",
        linewidth=2.5,
        label=r"Synthetic Maxima versus Gumbel Fit ($V_{\mathrm{syn}}$)",
    )

    ax_b.set_xlabel("Kuiper statistic")
    ax_b.set_ylabel("CDF")
    ax_b.set_ylim((0, 1.02))
    ax_b.legend(loc="lower right", fontsize=13)

    if xlim:
        ax_b.set_xlim((k_min, k_max))

    # panel labels, matching the convention in the other plotting modules but
    # placed outside the axes so neither collides with the schematic or curves
    for ax, label in ((ax_a, "A"), (ax_b_label, "B")):
        ax.text(
            -0.02,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONTSIZE,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    if save_figs:
        fname = make_figure_filename(*filename_args)
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {fname}")

    return fig, (ax_a, ax_b)
