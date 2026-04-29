# gev-heat-waves

A Python pipeline for fitting **Generalized Extreme Value (GEV)** distributions to observed and simulated heat wave data, aggregated over space. The project supports both stationary and non-stationary GEV models, applies Kuiper goodness-of-fit tests to assess fit quality, and is designed to run at scale on HPC clusters via MPI.

**By:** Adam Michael Bauer - University of Chicago

---

## Overview

This repository provides a full analysis pipeline for studying the statistical properties of heat wave extremes using Extreme Value Theory (EVT). The workflow covers:

1. **Preprocessing** — Land-masking, regridding, and anomaly computation for ERA5 reanalysis and CMIP6/AMIP climate model data.
2. **GEV Fitting** — Maximum likelihood estimation (MLE) of stationary and non-stationary GEV distributions at every grid point.
3. **Kuiper Testing** — Goodness-of-fit assessment comparing observed vs. GEV-implied distributions using the Kuiper statistic.
4. **Analysis & Visualization** — Jupyter notebooks for post-processing results and generating figures.

Data sources supported:
- **ERA5** reanalysis (1- and 0.5-degree grids)
- **CMIP6** multi-model ensembles (primary member or all members of the model with the most ensemble members)
- **AMIP** runs

Featured cities for extreme event analysis include Seattle, Mexico City, São Paulo, Lyon, Lagos, New Delhi, Moscow, Tokyo, and Melbourne.

---

## Repository Structure

```
gev-heat-waves/
├── src/evt_heat_waves/        # Core Python package
│   ├── config.py              # Path configuration and CLI argument parsers
│   ├── cmip_dataclass.py      # Dataclasses for CMIP6 ensemble metadata
│   ├── logging.py             # Logging setup
│   ├── utils.py               # Shared utility functions
│   ├── mle/                   # GEV MLE fitting routines
│   │   ├── mle.py             # Stationary & non-stationary GEV fitting via xarray
│   │   └── hess.py            # Hessian-based uncertainty estimation
│   ├── pproc/                 # Preprocessing modules
│   │   ├── pproc_era5.py      # ERA5 preprocessing (regrid, mask, anomalies)
│   │   ├── pproc_cmip.py      # CMIP6 preprocessing
│   │   └── pproc_amip.py      # AMIP preprocessing
│   ├── kuiper/                # Kuiper goodness-of-fit tests
│   │   ├── kuiper_fitting.py  # Kuiper statistic computation
│   │   └── bootstrap.py       # Bootstrapped Kuiper critical values
│   ├── mip_fit/               # CMIP/AMIP fitting orchestration
│   │   ├── cmip/              # CMIP-specific fitting routines
│   │   └── amip/              # AMIP-specific fitting routines
│   ├── check_plots/           # Diagnostic plots for QC
│   ├── main_pproc.py          # CLI entry point: preprocessing
│   ├── main_mip_fit.py        # CLI entry point: CMIP/AMIP GEV fitting
│   ├── main_era5_fitting.py   # ERA5-specific GEV fitting script
│   ├── main_fitting_mpi.py    # MPI-parallelized GEV fitting
│   └── main_kuiper_bootstrapping.py  # Kuiper bootstrap entry point
├── config/
│   ├── paths.yaml             # Data root and output path configuration (user-defined)
│   ├── meta.yaml              # CMIP6 ensemble metadata
│   ├── meta_amip.yaml         # AMIP ensemble metadata
│   ├── qc.yaml                # Quality control flags for CMIP6 models/years
│   ├── qc_amip.yaml           # QC flags for AMIP runs
│   ├── events_feat.yaml       # Featured city extreme event observations
│   └── events_all.yaml        # Full set of city-level extreme events
├── experiments/               # SLURM batch scripts for HPC execution
│   ├── era5_fitting.sbatch
│   ├── cmip_primary_mems_mpi.sbatch
│   ├── cmip_most_mems_mpi.sbatch
│   ├── amip_primary_mems_mpi.sbatch
│   ├── amip_most_mems_mpi.sbatch
│   ├── kuiper_mpi.sbatch
│   └── bootstrapped_kuiper.sbatch
├── analysis/
│   ├── notebooks/             # Jupyter notebooks for figures and analysis
│   └── scripts/               # Standalone analysis scripts
├── pyproject.toml             # Package metadata and CLI entry points
└── gev-heat-waves.yaml        # Conda environment specification
```

---

## Installation

### 1. Clone the repository

```bash
git clone -b refac https://github.com/adam-bauer-34/gev-heat-waves.git
cd gev-heat-waves
```

### 2. Create the conda environment

```bash
conda env create -f gev-heat-waves.yaml
conda activate gev-heat-waves
```

> **Note:** The environment file includes a local dependency on `ambpy` (a personal Python module). You will need to either install it from the specified path or remove/replace that dependency before installing.

### 3. Install the package

```bash
pip install -e .
```

### 4. Configure data paths

Create a `config/paths.yaml` file specifying where your data lives. The expected keys are:

```yaml
DATA_ROOT: /path/to/your/data
FIGS_PATH: /path/to/save/figures
ERA5_DIR:  ERA5        # subdirectory under DATA_ROOT
CMIP_DIR:  CMIP6       # subdirectory under DATA_ROOT
AMIP_DIR:  AMIP        # subdirectory under DATA_ROOT
STATS_DIR: stats       # subdirectory under DATA_ROOT
```

---

## Usage

The package exposes four CLI entry points after installation.

### Preprocessing

Preprocess ERA5, CMIP6, or AMIP data (land masking, regridding, anomaly computation):

```bash
pproc --data era5 --grid 1deg
pproc --data cmip
pproc --data amip
```

Options:
- `--data` — Data source: `era5`, `cmip`, or `amip` (default: `era5`)
- `--grid` — Grid resolution for ERA5: `1deg` or `0.5deg` (default: `1deg`)
- `--make_check_plots` — Generate diagnostic plots for land masking QC
- `--bypass-checks` — Skip manual YAML confirmation prompts
- `--debug` — Verbose logging; disables parallelization

### GEV Fitting (CMIP/AMIP)

Fit stationary or non-stationary GEV distributions to CMIP6 or AMIP data:

```bash
fit-mip --data cmip --fit nonstat --member_config prim
fit-mip --data amip --fit stat --member_config most --mpi
```

Options:
- `--data` — `cmip` or `amip` (default: `cmip`)
- `--fit` — Fit type: `stat`, `nonstat`, `stat_fixed_xi`, or `nonstat_fixed_xi_loc_only` (default: `nonstat`)
- `--member_config` — `prim` (primary member per model) or `most` (all members of the model with the most ensemble members) (default: `prim`)
- `--mpi` — Enable MPI parallelism for HPC execution
- `--debug` — Debug mode

### ERA5 GEV Fitting

Fit GEV distributions directly to ERA5 data:

```bash
python src/evt_heat_waves/main_era5_fitting.py GRID STAT [TMIN]
```

Arguments:
- `GRID` — Grid label (e.g., `1deg`)
- `STAT` — `stat` for stationary or `nonstat` for non-stationary
- `TMIN` (optional) — Start year for the analysis; defaults to the minimum year in the dataset

### Kuiper Bootstrap

Generate bootstrapped Kuiper critical values for goodness-of-fit testing:

```bash
bootstrap-kuiper --tmin 1979
```

Options:
- `--tmin` — Minimum year for the analysis interval (default: `1979`); sets the sample size as `2024 - tmin`
- `--debug` — Debug mode

---

## HPC / SLURM Execution

Pre-configured SLURM batch scripts are provided in `experiments/` for running the full pipeline on a cluster with MPI. Submit jobs with:

```bash
sbatch experiments/era5_fitting.sbatch
sbatch experiments/cmip_primary_mems_mpi.sbatch
sbatch experiments/cmip_most_mems_mpi.sbatch
sbatch experiments/kuiper_mpi.sbatch
sbatch experiments/bootstrapped_kuiper.sbatch
```

The MPI-enabled fitting scripts distribute tasks (variable × fit type × ensemble member combinations) across available processes using a round-robin scheme.

---

## Analysis Notebooks

Jupyter notebooks for post-processing and figure generation are in `analysis/notebooks/`:

| Notebook | Description |
|---|---|
| `viz_nonstat_era5_gev_analysis.ipynb` | Non-stationary GEV analysis for ERA5 |
| `viz_nonstat_cmip_gev_analysis.ipynb` | Non-stationary GEV analysis for CMIP6 |
| `viz_cmip_all_bias_corr.ipynb` | CMIP6 bias correction (all members) |
| `viz_cmip_primary_bias_corr.ipynb` | CMIP6 bias correction (primary members) |
| `viz_cmip_most_bias_corr.ipynb` | CMIP6 bias correction (most members model) |
| `viz_kuiper_analysis.ipynb` | Kuiper goodness-of-fit visualization |
| `viz_return_rates.ipynb` | Return period / return level analysis |
| `supp_mle_perf_stats.ipynb` | MLE performance and diagnostics |
| `supp_model_table.ipynb` | CMIP6 model summary table |

---

## GEV Model Details

The core fitting routine (`src/evt_heat_waves/mle/mle.py`) supports:

- **Stationary GEV** — three parameters: shape (ξ), location (μ), scale (σ)
- **Non-stationary GEV** — six parameters, with location and scale allowed to trend linearly in time
- Fits are performed at every (lat, lon) grid point using `scipy.optimize.minimize` with `xarray.apply_ufunc` for optional Dask parallelization
- Three temperature variables are fit for each dataset: raw annual maximum (`t2m`), anomaly relative to annual mean (`t2m_anom_annmean`), and anomaly relative to the trend in annual mean (`t2m_anom_trend`)

Goodness-of-fit is assessed via the **Kuiper statistic** (from `astropy.stats`), comparing the empirical CDF of observed annual maxima to the fitted GEV CDF. Bootstrapped critical values are generated to assess significance.

---

## Dependencies

Key dependencies (see `gev-heat-waves.yaml` for full list):

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy | 1.23.5 |
| xarray | 2022.11.0 |
| scipy | 1.10.1 |
| pandas | 1.5.3 |
| dask | latest |
| xesmf | latest |
| cartopy | latest |
| astropy | latest |
| matplotlib | 3.7.0 |
| seaborn | 0.12.2 |
| mpi4py | 4.1.1 |

---

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

---

## Contact

Adam Michael Bauer — ambauer [at] uchicago [dot] edu