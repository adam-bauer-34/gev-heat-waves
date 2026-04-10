#!/bin/bash

#SBATCH --job-name=kuiper_stats
#SBATCH --output=logs/kuiper_stats_%j.out
#SBATCH --error=logs/kuiper_stats_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =================================================================================
# Kuper Stats Fitting Job Script
# How to use:
#   1. set ntasks-per-node = # vars * # anomalies * # tmins
#   2. make sure you don't ask for too much time or you'll sit in queue forever!
# 
# Basic structure:
#   1. Make log directory if it doesn't exist and relevant outfiles.
#   2. Set up python (miniforge and ./.env) and MPI environments (using openmpi)
#   3. Run python script(s)
#   4. Report exit status and close.
# =================================================================================

# =================================================================================
# OUTFILE AND LOGGING
# =================================================================================

echo "============================================================"
echo "Starting CMIP6 GEV Fitting Job with MPI Parallelization"
echo "============================================================"
echo "Job started at: $(date)"
echo ""

# Ensure the logs directory exists
echo "Checking logs directory..."
mkdir -p logs
echo "Logs directory confirmed/created."
echo ""

# Create job information file
OUTFILE="logs/${SLURM_JOB_ID}.txt"
echo "Creating job information file: ${OUTFILE}"
touch "${OUTFILE}"

# pack a bunch of config info into outfile
{
    echo "Job ID: $SLURM_JOB_ID" 
    echo "Job name: $SLURM_JOB_NAME" 
    echo "Nodes: $SLURM_JOB_NUM_NODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total tasks: $SLURM_NTASKS"
    echo "CPUs per task: $SLURM_CPUS_PER_TASK"
    echo "Memory per CPU: $SLURM_MEM_PER_CPU MB"
    echo "Partition: $SLURM_JOB_PARTITION"
} > "${OUTFILE}"

echo "Job information saved to ${OUTFILE}"
echo ""

# =================================================================================
# CONFIG PYTHON AND MPI ENVIRONMENTS
# =================================================================================
echo "------------------------------------------------------------"
echo "Setting up Python and MPI environments"
echo "------------------------------------------------------------"

# Load MPI module and unload stuff that conflicts
module unload netcdf
module unload intelmpi 
module load openmpi

# Load Python module
echo "Loading Python module: python/miniforge-25.3.0"
module load python/miniforge-25.3.0

# Activate conda environment
echo "Activating conda environment: /project/bbcael/ambauer/envs/gev-heat-waves"
source activate /project/bbcael/ambauer/envs/gev-heat-waves

# Change to project directory
cd /project/bbcael/ambauer/gev-heat-waves

# Tell OpenMPI to use Slurm's process management
export OMPI_MCA_plm=slurm
export OMPI_MCA_btl_openib_allow_ib=0

# force MPI to use local ethernet ('tcp') instead of InfinitBand connections.
# raises error when I try to use InfinitiBand for reasons I don't understand.
# since this job doesn't talk much between jobs (embarassingly parallel workflow)
# I don't mind the slower inter-chip communication.
export OMPI_MCA_btl="tcp,self"

# Generate hostfile from Slurm allocation
export OMPI_MCA_orte_default_hostfile=$SLURM_JOB_NODELIST

# xarray reads these variables to determine how many threads to use when an operation
# natively does multi-threading. since cpu-per-task = 1 in our setup, tell xarray
# to only use 1 cpu per thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module list

echo "Environment setup complete."
echo ""

# display job config
echo "============================================================"
echo "Job Configuration Summary"
echo "============================================================"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "MPI tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total MPI tasks: $SLURM_NTASKS"
echo "Parallelism: Each fit 1 variable, anomaly type, and tmin"
echo "============================================================"
echo ""

# =============================================================================
# RUN PYTHON CODE USING MPIRUN
# NOTE: i'm not using srun here because, well, it didn't work the first time,
# and then i tried mpirun, and it worked, so i went with what worked. sorry?
# =============================================================================

echo "============================================================"
echo "Running parallelized Kuiper statistics script"
echo "============================================================"
echo "Starting at: $(date)"
echo ""

mpirun -np $SLURM_NTASKS \
    --mca plm slurm \
    --mca btl tcp,self \
    --mca btl_base_warn_component_unused 0 \
    python main_era5_fitting_kuiper_mpi.py 1deg

KUIPER_STATUS=0
echo ""
echo "Kuiper statistic completed at: $(date)"
echo "Exit status: ${STAT_STATUS}"
echo ""

# =============================================================================
# Job Summary
# =============================================================================

echo "============================================================"
echo "Job Summary"
echo "============================================================"
echo "Kuiper analysis exit status: ${KUIPER_STATUS}"
echo ""

if [ ${KUIPER_STATUS} -eq 0 ]; then
    echo "Runs completed successfully ✅"
else
    echo "Runs failed ❌"
fi

echo ""
echo "Job finished at: $(date)"
echo "============================================================"