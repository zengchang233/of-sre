#!/bin/sh

#SBATCH --job-name=resnet_stat_am_logfbank
#SBATCH --out="v100_resnet_stat_am_logfbank.out"
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:tesla_v100:1

source /home/smg/zengchang/.bashrc
module load cuda11.1
echo `which python`
echo `which nvcc`
nvcc -V
export CUDA_VISIBLE_DEVICES="0"

./run.sh
