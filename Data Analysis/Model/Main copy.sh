#!/bin/bash
#SBATCH --job-name=2AFC_Model_Fitting
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G              # Total memory (not per-cpu)
#SBATCH --time=48:00:00        # 48 hours
#SBATCH --output=Report_%A-%a.out
#SBATCH --error=Report_%A-%a.err
#SBATCH --mail-user=alishamsniaaa@gmail.com

# Load Anaconda module (if needed)
module load anaconda3

cd /storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/2AFC_computational_behavior_model
# Activate the virtual environment
conda activate suite2p_env

python main.py