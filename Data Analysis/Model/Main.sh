#!/bin/bash
#SBATCH --job-name=2AFC_Fit
#SBATCH --account=gts-fnajafi3
#SBATCH --partition=cpu-small    # 'cpu-small' is faster/easier to schedule than general
#SBATCH --array=0-47             # Runs the job 48 times (indices 0 to 47)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Give 4 dedicated cores to this job
#SBATCH --mem=8G                 # 8GB is plenty for one session
#SBATCH --time=72:00:00          # 4 hours should be plenty per session (likely takes 20-40 mins)
#SBATCH --output=Report_%A_%a.out  # Separate log for each session
#SBATCH --error=Report_%A_%a.err
#SBATCH --mail-user=alishamsniaaa@gmail.com
#SBATCH --mail-type=FAIL,END

# 1. Load Environment
module load anaconda3

cd /storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/2AFC_computational_behavior_model
# Activate the virtual environment
conda activate suite2p_env

# 2. CRITICAL: Bind threads to the allocated CPUs
# This prevents the "22 hours" issue by forcing PyTorch to stay on its 4 cores.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
python main.py