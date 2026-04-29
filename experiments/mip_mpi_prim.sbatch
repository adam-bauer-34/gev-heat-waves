#!/bin/bash

#SBATCH --job-name=cmip_primary
#SBATCH --output=experiments/logs/cmip_primary_%j.out
#SBATCH --error=experiments/logs/cmip_primary_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=33
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =================================================================================
# XMIP GEV Fitting Job Script
# How to use:
#   1. set ntasks-per-node = # of active models. this implies each task does
#      3 fitting routines (since there are two variables, daily max/min, which
#      sets the number of nodes).
#   2. make sure you don't ask for too much time or you'll sit in queue forever!
#   
#   NOTES:
#     - if CMIP, -ntasks-per-node=33 (5 tasks per node)
#     - if AMIP, -ntasks-per-node=36
#     - see README.md for more details
# 
# Basic structure:
#   1. Make log directory if it doesn't exist and relevant outfiles.
#   2. Set up python (miniforge and ./.env) and MPI environments (using openmpi)
#   3. Run python script
#   4. Report exit status and close.
# =================================================================================

# =================================================================================
# PARSE CLI ARGUMENTS
# =================================================================================

DATA_ARG=""
FIT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA_ARG="$2"
            shift 2
            ;;
        --fit)
            FIT_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch $0 --data <arg> --fit <arg>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_ARG" ]]; then
    echo "ERROR: --data argument is required."
    echo "Usage: sbatch $0 --data <arg> --fit <arg>"
    exit 1
fi

if [[ -z "$FIT_ARG" ]]; then
    echo "ERROR: --fit argument is required."
    echo "Usage: sbatch $0 --data <arg> --fit <arg>"
    exit 1
fi

# =================================================================================
# OUTFILE AND LOGGING
# =================================================================================

echo "============================================================"
echo "Starting CMIP6 GEV Fitting Job with MPI Parallelization"
echo "============================================================"
echo "Job started at: $(date)"
echo "Arguments: --data ${DATA_ARG} --fit ${FIT_ARG}"
echo ""

# Ensure the logs directory exists
echo "Checking logs directory..."
mkdir -p experiments/logs
echo "Logs directory confirmed/created."
echo ""

# Create job information file
OUTFILE="experiments/logs/${SLURM_JOB_ID}.txt"
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
    echo "CLI Arguments: --data ${DATA_ARG} --fit ${FIT_ARG}"
    echo ""
    echo "Strategy: Each of 3 fits per model runs as independent MPI task"
    echo "Maximum parallelism: $(($SLURM_NTASKS / 3)) fits can be processed simultaneously"
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

echo "============================================================"
echo "Running main file"
echo "============================================================"
echo "Command: mpirun [mpi-config-stuff] python -m src.evt_heat_waves.mip_fit.main_mpi_prim --data ${DATA_ARG} --fit ${FIT_ARG}"
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
    python -m src.evt_heat_waves.mip_fit.main_mpi_prim --data "${DATA_ARG}" --fit "${FIT_ARG}"

# record status of previous run
STATUS=$?

echo ""
echo "Main task completed at: $(date)"
echo "Exit status: ${STATUS}"
echo ""