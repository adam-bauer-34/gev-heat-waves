#!/bin/bash

#SBATCH --job-name=kuiper_bootstrap
#SBATCH --output=experiments/logs/kuiper_bootstrap_%j.out
#SBATCH --error=experiments/logs/kuiper_bootstrap_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# =================================================================================
# Kuper Bootstrapping Fitting Job Script
# 
# Basic structure:
#   1. Make log directory if it doesn't exist and relevant outfiles.
#   2. Set up python (miniforge and ./.env)
#   3. Run python script(s)
#   4. Report exit status and close.
# =================================================================================

# =================================================================================
# OUTFILE AND LOGGING
# =================================================================================

echo "============================================================"
echo "Starting Kuiper bootstrapping job"
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
} > "${OUTFILE}"

echo "Job information saved to ${OUTFILE}"
echo ""

# =================================================================================
# CONFIG PYTHON AND MPI ENVIRONMENTS
# =================================================================================
echo "------------------------------------------------------------"
echo "Setting up Python and MPI environments"
echo "------------------------------------------------------------"

# Load Python module
echo "Loading Python module: python/miniforge-25.3.0"
module load python/miniforge-25.3.0

# Activate conda environment
echo "Activating conda environment: /project/bbcael/ambauer/envs/gev-heat-waves"
source activate /project/bbcael/ambauer/envs/gev-heat-waves

# Change to project directory
cd /project/bbcael/ambauer/gev-heat-waves

# =============================================================================
# RUN PYTHON CODE USING MPIRUN
# NOTE: i'm not using srun here because, well, it didn't work the first time,
# and then i tried mpirun, and it worked, so i went with what worked. sorry?
# =============================================================================

echo "============================================================"
echo "Running bootstrapping script"
echo "============================================================"
echo "Starting at: $(date)"
echo ""

python main_kuiper_bootstrapping.py

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