#!/bin/bash

#SBATCH --job-name=cmip_primary
#SBATCH --output=logs/cmip_primary_%j.out
#SBATCH --error=logs/cmip_primary_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =================================================================================
# CMIP GEV Fitting Job Script
# How to use:
#   1. set ntasks-per-node = # of active models. this implies each task does
#      3 fitting routines (since there are two variables, daily max/min, which
#      sets the number of nodes).
#   2. make sure you don't ask for too much time or you'll sit in queue forever!
# 
# Basic structure:
#   1. Make log directory if it doesn't exist and relevant outfiles.
#   2. Set up python (miniforge and ./.env) and MPI environments (using openmpi)
#   3. Run python script(s)
#       * NOTE: remember to change the time you ask for if you're doing `stat` and
#         `nonstat` fits versus only one!
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
    echo ""
    echo "Strategy: Each of 3 fits per model runs as independent MPI task"
    echo "Maximum parallelism: $(($SLURM_NTASKS / 3)) models can be processed simultaneously"
} > "${OUTFILE}"

echo "Job information saved to ${OUTFILE}"
echo ""

# =================================================================================
# CONFIG PYTHON AND MPI ENVIRONMENTS
# =================================================================================
echo "------------------------------------------------------------"
echo "Setting up Python and MPI environments"
echo "------------------------------------------------------------"

# Load MPI module
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
export OMPI_MCA_btl="tcp,self"  # use local ethernet
export OMPI_MCA_coll="^hcoll"  # get rid of noisy warnings
export OMPI_MCA_pml="^ucx"  # get rid of noisy warnings

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
echo "Parallelism: Each fit (3) for each variable (2) is an independent MPI task"
echo "Max models processed simultaneously: $(($SLURM_NTASKS / 6))"
echo "============================================================"
echo ""

# =============================================================================
# RUN PYTHON CODE USING MPIRUN
# NOTE: i'm not using srun here because, well, it didn't work the first time,
# and then i tried mpirun, and it worked, so i went with what worked. sorry?
# =============================================================================

#echo "============================================================"
#echo "Running STATIONARY GEV Fitting"
#echo "============================================================"
#echo "Command: srun python main_cmip_fitting_primary_mems_mpi.py stat"
#echo "Starting at: $(date)"
#echo ""

#mpirun -np $SLURM_NTASKS \
#    --mca plm slurm \
#    --mca btl tcp,self \
#    --mca btl_base_warn_component_unused 0 \
#    python main_cmip_fitting_primary_mems_mpi.py stat

STAT_STATUS=0
echo ""
echo "Stationary fitting completed at: $(date)"
echo "Exit status: ${STAT_STATUS}"
echo ""

echo "============================================================"
echo "Running NON-STATIONARY GEV Fitting"
echo "============================================================"
echo "Command: mpirun [mpi-config-stuff] python main_cmip_fitting_primary_mems_mpi.py nonstat"
echo "Starting at: $(date)"
echo ""

# use mpirun to parallelize the code
# first arguement tells mpirun how many tasks to do
# set warn component unused because i hard set the code to use ethernet,
# which is fine since i have minimal crosstalk between fitting routines
# (i.e., my workflow is embarassingly parallel)
mpirun -np $SLURM_NTASKS \
    --mca plm slurm \
    --mca btl tcp,self \
    --mca btl_base_warn_component_unused 0 \
    python main_cmip_fitting_primary_mems_mpi.py nonstat


# record status of previous run
NONSTAT_STATUS=$?

echo ""
echo "Non-stationary fitting completed at: $(date)"
echo "Exit status: ${NONSTAT_STATUS}"
echo ""

# =============================================================================
# Job Summary
# =============================================================================

echo "============================================================"
echo "Job Summary"
echo "============================================================"
echo "Stationary fit exit status: ${STAT_STATUS}"
echo "Non-stationary fit exit status: ${NONSTAT_STATUS}"
echo ""

if [ ${STAT_STATUS} -eq 0 ] && [ ${NONSTAT_STATUS} -eq 0 ]; then
    echo "Status: All runs completed successfully ✅"
else
    echo "Status: One or more runs failed ❌"
    echo ""
    echo "Failed runs:"
    [ ${STAT_STATUS} -ne 0 ] && echo "  - Stationary fitting"
    [ ${NONSTAT_STATUS} -ne 0 ] && echo "  - Non-stationary fitting"
fi

echo ""
echo "Job finished at: $(date)"
echo "============================================================"