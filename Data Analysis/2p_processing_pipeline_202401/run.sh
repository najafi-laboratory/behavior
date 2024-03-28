#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=12:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/2p_processing_pipeline_202401
source activate suite2p
python run_suite2p_pipeline.py \
--denoise 0 \
--spatial_scale 1 \
--data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/FN16/032124/FN16P_Omi_W_032124-346' \
--save_path './results/FN16_P_omi_032124_w' \
--nchannels 2 \
--functional_chan 2 \
--brain_region 'ppc' \