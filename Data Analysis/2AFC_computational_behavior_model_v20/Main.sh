#!/bin/bash
#SBATCH --job-name=2AFC_Fit_MC06
#SBATCH --account=gts-fnajafi3
#SBATCH --partition=cpu-small    
#SBATCH --array=0-47           # Adjust based on the number of sessions in your list
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=8G                 
#SBATCH --time=90:00:00          
#SBATCH --output=Report_%A_%a.out  
#SBATCH --error=Report_%A_%a.err
#SBATCH --mail-user=alishamsniaaa@gmail.com
#SBATCH --mail-type=FAIL,END

module load anaconda3
cd /storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/2afc_computational_behavior_model_v20
conda activate suite2p_env

# CRITICAL: Bind threads to the allocated CPUs to prevent node hanging/contention
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configure these once for the whole array job.
# DATA_PATHS_FILE should contain one .mat path per line in the same order as the array indices.
export DATA_PATHS_FILE=${DATA_PATHS_FILE:-/storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/2afc_computational_behavior_model_v20/data_paths.txt}
export SAVE_PATH=${SAVE_PATH:-/storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/results}

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
python main.py
