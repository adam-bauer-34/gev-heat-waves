#!/bin/bash

#SBATCH --job-name=era5_kuiper
#SBATCH --output=experiments/logs/era5_kuiper_%j.out
#SBATCH --error=experiments/logs/era5_kuiper_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =================================================================================
# ERA5 GEV Fitting Job Script
# How to use:
#   1. set ntasks-per-node = N_anomaly_types * N_tmins * N_data_types (= 12 in general)
#   2. make sure you don't ask for too much time or you'll sit in queue forever!
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

FIT_ARG=""
GRID_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fit)
            FIT_ARG="$2"
            shift 2
            ;;
        --grid)
            GRID_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch $0 --fit <arg> --grid <arg>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FIT_ARG" ]]; then
    echo "ERROR: --fit argument is required."
    echo "Usage: sbatch $0 --fit <arg> --grid <arg>"
    exit 1
fi

if [[ -z "$GRID_ARG" ]]; then
    echo "ERROR: --grid argument is required."
    echo "Usage: sbatch $0 --fit <arg> --grid <arg>"
    exit 1
fi

# =================================================================================
# OUTFILE AND LOGGING
# =================================================================================

echo "============================================================"
echo "Starting ERA5 GEV Fitting Job with MPI Parallelization"
echo "============================================================"
echo "Job started at: $(date)"
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
    echo "CLI Arguments: --fit ${FIT_ARG} --grid ${GRID_ARG}"
    echo ""
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
echo "Command: mpirun [mpi-config-stuff] python -m src.evt_heat_waves.era5.main_mpi_kuiper --fit ${FIT_ARG} --grid ${GRID_ARG}"
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
    python -m src.evt_heat_waves.era5.main_mpi_kuiper --fit "${FIT_ARG}" --grid "${GRID_ARG}"

# record status of previous run
STATUS=$?

echo ""
echo "Main task completed at: $(date)"
echo "Exit status: ${STATUS}"
echo ""