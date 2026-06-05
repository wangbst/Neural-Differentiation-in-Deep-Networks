#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

source ~/.bashrc

conda activate PyTorch


export TORCH_HOME="your path"

cd "your path"
python train.py
