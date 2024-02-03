#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=6:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/2p_processing_pipeline_202401
source activate suite2p
python main.py \
--run_Registration 1 \
--run_CellDetect 1 \
--run_Extraction 1 \
--run_SyncSignal 1 \
--run_RetrieveResults 1 \
--run_Plotter 1 \
--data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/FN8/02022024/FN8_P_Omi_020224_788_1113_-298-268' \
--save_path0 './results/FN8_020224_omission' \
--nchannels 2 \
--functional_chan 2 \
--diameter 16