"""Main file for GEV fitting of ERA5 data.

Adam Michael Bauer
UChicago
11.10.2025

To run: main_gev_fit.py GRID STAT [TMIN]

Last edited: 2/6/2026, 11:02 AM CST
"""

import sys
import shutil

import xarray as xr

from config import DATA_ROOT
from src.mle import ds_mle_fit, reset_mle_stats
from pathlib import Path

# import command line arguments
GRID = sys.argv[1]
STAT = sys.argv[2]

# Check if TMIN was provided
if len(sys.argv) > 3:
    TMIN = int(sys.argv[3])
    tmin_specified = True
else:
    TMIN = None
    tmin_specified = False

# terminal width
width = shutil.get_terminal_size(fallback=(80, 20)).columns

print("=" * width)
print("🌍 ERA5 GEV Fitting Pipeline")
print("=" * width)
print(f"📊 Configuration:")
print(f"-" * width)
print(f"   Grid: {GRID}")
print(f"   Statistical Model: {STAT}")
if tmin_specified:
    print(f"   TMIN: {TMIN} (user-specified)")
else:
    print(f"   TMIN: Will use dataset minimum")
print("=" * width)

print("\n📂 Importing land-masked data...")
# define variables and open datasets
vars = ['t2m_annual_max']
fnames = ['era5_' + var + '_' + GRID + '_landonly.nc' for var in vars]
print(f"   Variables: {', '.join(vars)}")
print(f"   Loading {len(fnames)} datasets from {DATA_ROOT / 'ERA5' / 'landonly'}")

dss = [xr.open_dataset(DATA_ROOT / 'ERA5' / 'landonly' / fname) for fname in fnames]

# If TMIN not provided, extract minimum year from first dataset
if TMIN is None:
    TMIN = int(dss[0]['year'].min().values)
    print(f"   📅 Using dataset minimum year as TMIN: {TMIN}")

# Slice datasets based on TMIN
print(f"   ✂️  Slicing data from year {TMIN} to {int(dss[0]['year'].max().values)}")
dss = [ds.sel(year=slice(TMIN, None)) for ds in dss]
print("✅ Data successfully loaded and sliced!\n")

# carry out GEV fitting for each dataset
if STAT == 'stat':
    print("🔧 Performing STATIONARY GEV fits...")
    print("   Step 1/3: Fitting GEV to 't2m' variable...")
    dss_with_fit = [ds_mle_fit(ds, var_name='t2m') for ds in dss]
    reset_mle_stats()
    
    print("   Step 2/3: Fitting GEV to 't2m_anom_annmean' variable...")
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom_annmean') for ds in dss_with_fit]
    reset_mle_stats()
    
    print("   Step 3/3: Fitting GEV to 't2m_anom_trend' variable...")
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom_trend') for ds in dss_with_fit_on_both]
    reset_mle_stats()

elif STAT == 'nonstat':
    print("🔧 Performing NON-STATIONARY GEV fits...")
    print("   Step 1/3: Fitting non-stationary GEV to 't2m' variable...")
    dss_with_fit = [ds_mle_fit(ds, var_name='t2m',
                               fit_dim='year', non_stat=True) for ds in dss]
    reset_mle_stats()
    
    print("   Step 2/3: Fitting non-stationary GEV to 't2m_anom_annmean' variable...")
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom_annmean',
                                       fit_dim='year', non_stat=True) for ds in dss_with_fit]
    reset_mle_stats()
    
    print("   Step 3/3: Fitting non-stationary GEV to 't2m_anom_trend' variable...")
    dss_with_fit_on_both = [ds_mle_fit(ds, var_name='t2m_anom_trend',
                                       fit_dim='year', non_stat=True) for ds in dss_with_fit_on_both]
    reset_mle_stats()

else:
    print("❌ Error: Invalid STAT argument!")
    raise ValueError("Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
    
print("✅ GEV fitting complete!\n")

print("\n💾 Saving datasets...")
# Create output directory if it doesn't exist
output_dir = DATA_ROOT / 'ERA5' / 'gev'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"   Output directory: {output_dir}")

# save datasets
for i, (VAR, ds_masked) in enumerate(zip(vars, dss_with_fit_on_both), 1):
    output_filename = f'era5_{VAR}_{GRID}_landonly_gev_{STAT}_TMIN{TMIN}.nc'
    output_path = output_dir / output_filename
    print(f"   [{i}/{len(vars)}] Saving {output_filename}...")
    ds_masked.to_netcdf(output_path)

print(f"✅ All datasets saved to {output_dir}")
print("\n" + "=" * width)
print("🎉 Pipeline complete!")
print("=" * width)