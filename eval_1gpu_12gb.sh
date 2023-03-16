#!/bin/bash

#SBATCH --partition=kate_reserved
#SBATCH --exclude matrix-1-[14,22,24],matrix-2-[25,29],matrix-0-36
#SBATCH --job-name=analogical-grasping
#SBATCH --output=slurm_logs/analogical-grasping-%j.out
#SBATCH --error=slurm_logs/analogical-grasping-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50gb

python gcngrasp/eval.py "$@"
