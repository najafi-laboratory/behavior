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
--data_path '/storage/coda1/p-fnajafi3/0/shared/2P_Imaging/FN8/12132023/Tseries FN8_PPC_121323_-504_704_219-213' \
--save_path0 'FN8_PPC_121323' \
--functional_chan 2