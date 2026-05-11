Last edited: 4/30/2026, 7:40 PM CST

# GEV Heat Waves

A comprehensive Python pipeline for fitting **Generalized Extreme Value (GEV)** distributions to heat wave extremes in observational and climate model data. Supports stationary and non-stationary models, Kuiper goodness-of-fit testing, and HPC-scale parallelization via MPI.

**Author:** Adam Michael Bauer, University of Chicago

---

## Overview

This project implements an end-to-end extreme value analysis pipeline for studying heat wave statistics using GEV models. The workflow includes:

- **Preprocessing** — Land masking, regridding, and anomaly computation for ERA5, CMIP6, and AMIP data
- **GEV Fitting** — Maximum likelihood estimation of stationary and non-stationary distributions at every grid point
- **Goodness-of-Fit Testing** — Kuiper statistic-based assessment and bootstrapped significance testing
- **Analysis & Visualization** — Jupyter notebooks for post-processing and figure generation

Supported datasets:
- **ERA5** reanalysis (1° and 0.5° resolution)
- **CMIP6** multi-model ensembles (primary members or all members from the ensemble-richest model)
- **AMIP** atmosphere-only runs

Three temperature variables are analyzed: raw annual maxima, deviations from annual mean, and deviations from the time-varying trend.

## Project Structure

```
gev-heat-waves/
├── src/evt_heat_waves/          # Core package
│   ├── config.py                # Configuration & CLI utilities
│   ├── logging_utils.py         # Logging configuration
│   ├── utils.py                 # Shared utilities
│   ├── mle/                     # GEV MLE fitting
│   │   ├── mle.py               # Core MLE solver
│   │   ├── hess.py              # Hessian-based uncertainty
│   │   ├── grad.py              # Gradient computation
│   │   ├── se.py                # Standard error estimation
│   │   └── utils.py             # MLE utilities
│   ├── plotting/                # Diagnostic visualization
│   │   ├── check_plots.py       # QC plots
│   │   └── plotting_presets.py  # Plot configuration
│   ├── pproc/                   # Preprocessing pipeline
│   │   ├── cli.py               # CLI interface
│   │   ├── main.py              # Main entry point
│   │   ├── preprocessing.py     # Core preprocessing
│   │   ├── pproc_era5.py        # ERA5 preprocessing
│   │   ├── pproc_cmip.py        # CMIP6 preprocessing
│   │   └── pproc_amip.py        # AMIP preprocessing
│   ├── era5/                    # ERA5-specific analysis
│   │   ├── cli.py               # CLI interface
│   │   ├── fit/                 # Fitting routines
│   │   ├── kuiper/              # Kuiper goodness-of-fit
│   │   │   ├── kuiper_fitting.py
│   │   │   ├── bootstrap.py
│   │   │   ├── kuipers.py
│   │   │   └── kuipers_mpi.py
│   │   ├── main_fit.py          # Serial fitting
│   │   ├── main_mpi_fit.py      # MPI fitting
│   │   ├── main_kuiper.py       # Kuiper test
│   │   ├── main_mpi_kuiper.py   # MPI Kuiper test
│   │   └── main_kuiper_bootstrapping.py  # Bootstrap routine
│   └── mip_fit/                 # CMIP/AMIP orchestration
│       ├── cli.py               # CLI interface
│       ├── cmip_dataclass.py    # CMIP6/AMIP metadata
│       ├── main_serial.py       # Serial fitting
│       ├── main_mpi_prim.py     # MPI (primary members)
│       ├── main_mpi_most.py     # MPI (most members)
│       ├── prim/                # Primary member routines
│       └── most/                # Most members routines
├── config/                      # Configuration files
│   ├── paths.yaml               # Data paths (user-configured)
│   ├── meta.yaml                # CMIP6 metadata
│   ├── meta.generated.yaml      # Generated CMIP6 metadata
│   ├── meta_amip.yaml           # AMIP metadata
│   ├── meta_amip.generated.yaml # Generated AMIP metadata
│   ├── qc.yaml                  # CMIP6 quality control flags
│   ├── qc.generated.yaml        # Generated QC flags
│   ├── qc_amip.yaml             # AMIP quality control flags
│   ├── mle_attrs.yaml           # MLE attribute specifications
│   ├── events_feat.yaml         # Featured city extremes
│   └── events_all.yaml          # All city-level extremes
├── experiments/                 # SLURM job submission scripts
│   ├── era5_mpi_fit.sbatch
│   ├── era5_mpi_kuiper.sbatch
│   ├── cmip_most_mems_mpi.sbatch
│   ├── cmip_primary_mems_mpi.sbatch
│   ├── amip_most_mems_mpi.sbatch
│   ├── amip_primary_mems_mpi.sbatch
│   └── bootstrapped_kuiper.sbatch
├── analysis/
│   ├── notebooks/               # Jupyter analysis notebooks
│   ├── scripts/                 # Standalone Python scripts
│   └── dev/                     # Development notebooks
├── pyproject.toml               # Package metadata & CLI entry points
├── gev-heat-waves.yaml          # Conda environment specification
└── README.md                    # This file
```

## Installation

**Requirements:** Python 3.9+, conda, and (for HPC use) an MPI installation.

1. **Clone the repository:**
   ```bash
   git clone -b refac https://github.com/adam-bauer-34/gev-heat-waves.git
   cd gev-heat-waves
   ```

2. **Create and activate the environment:**
   ```bash
   conda env create -f gev-heat-waves.yaml
   conda activate gev-heat-waves
   ```
   
   > **Note:** The environment file specifies a local `ambpy` dependency. Install it from the specified path or remove the dependency before proceeding.

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Configure data paths:**
   
   Create `config/paths.yaml` with your data locations:
   ```yaml
   DATA_ROOT: /path/to/data          # Root directory for all data
   FIGS_PATH: /path/to/figures       # Output directory for figures
   ERA5_DIR:  ERA5                   # Subdirectory for ERA5 data
   CMIP_DIR:  CMIP6                  # Subdirectory for CMIP6 data
   AMIP_DIR:  AMIP                   # Subdirectory for AMIP data
   STATS_DIR: stats                  # Subdirectory for outputs
   ```

## Quick Start

### Preprocessing

Preprocess ERA5, CMIP6, or AMIP data (land masking, regridding, anomaly computation):

```bash
pproc --data era5 --grid 1deg
pproc --data cmip
pproc --data amip
```

**Options:**
- `--data` — Data source: `era5`, `cmip`, `amip` (default: `era5`)
- `--grid` — ERA5 resolution: `1deg`, `0.5deg` (default: `1deg`)
- `--make_check_plots` — Generate diagnostic plots for QC
- `--bypass-checks` — Skip confirmation prompts
- `--debug` — Verbose logging; disables parallelization

### GEV Fitting (CMIP/AMIP)

Fit stationary or non-stationary GEV distributions:

```bash
fit-mip --data cmip --fit nonstat --member_config prim
fit-mip --data amip --fit stat --member_config most --mpi
```

**Options:**
- `--data` — `cmip` or `amip` (default: `cmip`)
- `--fit` — `stat`, `nonstat`, `stat_fixed_xi`, `nonstat_fixed_xi_loc_only` (default: `nonstat`)
- `--member_config` — `prim` (primary) or `most` (richest ensemble) (default: `prim`)
- `--mpi` — Enable MPI parallelization
- `--debug` — Debug mode

### ERA5 GEV Fitting

Fit GEV distributions to ERA5 data:

```bash
fit-era5 1deg nonstat 1979
mpi-fit-era5 1deg stat
```

**Arguments:**
- Grid resolution (e.g., `1deg`, `0.5deg`)
- Fit type: `stat` or `nonstat`
- Start year (optional; defaults to dataset minimum)

### Kuiper Goodness-of-Fit

Generate bootstrapped Kuiper critical values:

```bash
bootstrap-kuiper --tmin 1979
```

**Options:**
- `--tmin` — Minimum year; sets sample size as 2024 − tmin (default: `1979`)
- `--debug` — Debug mode

## HPC / SLURM Execution

Pre-configured SLURM job scripts are provided in `experiments/` for cluster deployment with MPI. Submit with:

```bash
sbatch experiments/era5_fitting.sbatch
sbatch experiments/cmip_primary_mems_mpi.sbatch
sbatch experiments/cmip_most_mems_mpi.sbatch
sbatch experiments/kuiper_mpi.sbatch
sbatch experiments/bootstrapped_kuiper.sbatch
```

MPI-enabled fitting distributes variable × fit type × ensemble member combinations across available processes using a round-robin scheme.

## Analysis & Visualization

Post-processing and figure generation notebooks are located in `analysis/notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `viz_nonstat_era5_gev_analysis.ipynb` | Non-stationary GEV analysis (ERA5) |
| `viz_nonstat_cmip_gev_analysis.ipynb` | Non-stationary GEV analysis (CMIP6) |
| `viz_cmip_all_bias_corr.ipynb` | CMIP6 bias correction (all members) |
| `viz_cmip_primary_bias_corr.ipynb` | CMIP6 bias correction (primary members) |
| `viz_cmip_most_bias_corr.ipynb` | CMIP6 bias correction (ensemble-richest model) |
| `viz_kuiper_analysis.ipynb` | Kuiper test visualization and diagnostics |
| `viz_return_rates.ipynb` | Return period and return level analysis |
| `supp_mle_perf_stats.ipynb` | MLE convergence and diagnostic statistics |
| `supp_model_table.ipynb` | CMIP6 model metadata summary |

## Technical Details

### GEV Model

The core fitting engine (`src/evt_heat_waves/mle/mle.py`) implements:

- **Stationary GEV** — Three parameters: shape (ξ), location (μ), scale (σ)
- **Non-stationary GEV** — Six parameters with location and scale trending linearly in time
- **Optimization** — Maximum likelihood estimation via `scipy.optimize.minimize` with `xarray.apply_ufunc` for Dask-based parallelization
- **Temperature variables** — Three variables fit independently at each grid point:
  - Raw annual maxima (`t2m`)
  - Anomaly relative to annual mean (`t2m_anom_annmean`)
  - Anomaly relative to time-varying trend (`t2m_anom_trend`)

### Goodness-of-Fit Testing

The **Kuiper statistic** (from `astropy.stats`) compares the empirical CDF of observed annual maxima to the fitted GEV CDF. Bootstrapped critical values enable significance assessment.

### Dependencies

See `gev-heat-waves.yaml` for the full environment specification. Key packages:

- **Core:** Python 3.9+, numpy, xarray, scipy, pandas
- **Geospatial:** xesmf, cartopy
- **Parallelization:** dask, mpi4py
- **Analysis:** astropy, matplotlib, seaborn

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@software{bauer_2025_gev_heat_waves,
  author = {Bauer, Adam Michael},
  title = {GEV Heat Waves: Extreme Value Analysis of Heat Wave Extremes},
  url = {https://github.com/adam-bauer-34/gev-heat-waves},
  year = {2026}
}
```

## Contact

**Adam Michael Bauer**  
University of Chicago  
ambauer [at] uchicago [dot] edu