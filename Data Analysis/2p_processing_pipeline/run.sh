#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=384G
#SBATCH --time=4:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/2p_processing_pipeline_202401
source activate suite2p
python main.py \
--data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/FN8/01312024/FN8_P_013124_41_477_3-264' \
--save_path0 './results/FN6_PPC_013124_passive' \
--nchannels 2 \
--functional_chan 2 \
--diameter 0