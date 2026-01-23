#!/bin/bash

#SBATCH --job-name=era5_fitting
#SBATCH --output=logs/era5_fitting.out
#SBATCH --error=logs/era5_fitting.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=24:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=4000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =============================================================================
# ERA5 GEV Fitting Job Script
# =============================================================================

echo "============================================================"
echo "Starting ERA5 GEV Fitting Job"
echo "============================================================"
echo "Job started at: $(date)"
echo ""

# ensure the logs directory actually exists
echo "Checking logs directory..."
mkdir -p logs
echo "Logs directory confirmed/created."
echo ""

# name and make the outfile
OUTFILE="logs/${SLURM_JOB_ID}.txt"
echo "Creating job information file: ${OUTFILE}"
touch "${OUTFILE}"

{
	echo "Job ID: $SLURM_JOB_ID" 
	echo "Job name: $SLURM_JOB_NAME" 
	echo "N tasks: $SLURM_ARRAY_TASK_COUNT" 
	echo "N cores: $SLURM_CPUS_ON_NODE" 
	echo "N threads per code: $SLURM_THREADS_PER_CORE" 
	echo "Minimum memory required per CPU: $SLURM_MEM_PER_CPU" 
	echo "Requested memory per GPU: $SLURM_MEM_PER_GPU" 
} > "${OUTFILE}"

echo "Job information saved to ${OUTFILE}"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

echo "------------------------------------------------------------"
echo "Setting up Python environment"
echo "------------------------------------------------------------"

# activate env
echo "Loading Python module: python/miniforge-25.3.0"
module load python/miniforge-25.3.0

echo "Activating conda environment: /project/bbcael/ambauer/gev-heat-waves/.env"
source activate /project/bbcael/ambauer/gev-heat-waves/.env

echo "Environment setup complete."
echo ""

# =============================================================================
# Running Python Scripts - Stationary Fits
# =============================================================================

echo "============================================================"
echo "Running STATIONARY GEV Fitting Analyses"
echo "============================================================"
echo ""

# Run stationary fits with 1979 being first year of analysis
echo "------------------------------------------------------------"
echo "Run 1: Stationary fit with 1979 as first year"
echo "------------------------------------------------------------"
echo "Command: python main_era5_fitting.py 1deg stat 1979"
echo "Starting at: $(date)"
echo ""

python main_era5_fitting.py 1deg stat 1979

RUN1_STATUS=$?
echo ""
echo "Run 1 completed at: $(date)"
echo "Exit status: ${RUN1_STATUS}"
echo ""

# Run stationary fits with the first year of data as first year of analysis
echo "------------------------------------------------------------"
echo "Run 2: Stationary fit with default first year"
echo "------------------------------------------------------------"
echo "Command: python main_era5_fitting.py 1deg stat"
echo "Starting at: $(date)"
echo ""

python main_era5_fitting.py 1deg stat

RUN2_STATUS=$?
echo ""
echo "Run 2 completed at: $(date)"
echo "Exit status: ${RUN2_STATUS}"
echo ""

# =============================================================================
# Running Python Scripts - Non-stationary Fits
# =============================================================================

echo "============================================================"
echo "Running NON-STATIONARY GEV Fitting Analyses"
echo "============================================================"
echo ""

# Run non-stationary fits with 1979 being first year of analysis
echo "------------------------------------------------------------"
echo "Run 3: Non-stationary fit with 1979 as first year"
echo "------------------------------------------------------------"
echo "Command: python main_era5_fitting.py 1deg nonstat 1979"
echo "Starting at: $(date)"
echo ""

python main_era5_fitting.py 1deg nonstat 1979

RUN3_STATUS=$?
echo ""
echo "Run 3 completed at: $(date)"
echo "Exit status: ${RUN3_STATUS}"
echo ""

# Run non-stationary fits with the first year of data as first year of analysis
echo "------------------------------------------------------------"
echo "Run 4: Non-stationary fit with default first year"
echo "------------------------------------------------------------"
echo "Command: python main_era5_fitting.py 1deg nonstat"
echo "Starting at: $(date)"
echo ""

python main_era5_fitting.py 1deg nonstat

RUN4_STATUS=$?
echo ""
echo "Run 4 completed at: $(date)"
echo "Exit status: ${RUN4_STATUS}"
echo ""

# =============================================================================
# Job Summary
# =============================================================================

echo "============================================================"
echo "Job Summary"
echo "============================================================"
echo "Run 1 (stationary, 1979 start) exit status: ${RUN1_STATUS}"
echo "Run 2 (stationary, default start) exit status: ${RUN2_STATUS}"
echo "Run 3 (non-stationary, 1979 start) exit status: ${RUN3_STATUS}"
echo "Run 4 (non-stationary, default start) exit status: ${RUN4_STATUS}"
echo ""

if [ ${RUN1_STATUS} -eq 0 ] && [ ${RUN2_STATUS} -eq 0 ] && [ ${RUN3_STATUS} -eq 0 ] && [ ${RUN4_STATUS} -eq 0 ]; then
    echo "Status: All runs completed successfully"
else
    echo "Status: One or more runs failed"
    echo ""
    echo "Failed runs:"
    [ ${RUN1_STATUS} -ne 0 ] && echo "  - Run 1 (stationary, 1979 start)"
    [ ${RUN2_STATUS} -ne 0 ] && echo "  - Run 2 (stationary, default start)"
    [ ${RUN3_STATUS} -ne 0 ] && echo "  - Run 3 (non-stationary, 1979 start)"
    [ ${RUN4_STATUS} -ne 0 ] && echo "  - Run 4 (non-stationary, default start)"
fi

echo ""
echo "Job finished at: $(date)"
echo "============================================================"