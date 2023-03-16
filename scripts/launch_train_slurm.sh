#!/bin/sh

if [[ $# -ne 3 ]]
then
  echo "usage: sh launch_train_slurm.sh <name_prefix> <model> <split_mode>"
  echo "example: sh launch_train_slurm.sh change_layer_num_42 sgn o"
  exit
fi

name_prefix=$1
model=$2
split_mode=$3

for split_idx in 0 1 2 3; do
  checkpoint_name="${name_prefix}_${model}_${split_mode}_${split_idx}"
  echo "checkpoint name: ${checkpoint_name}_<timestamp>"
  sbatch train_1gpu_12gb.sh \
    --cfg_file cfg/train/${model}/${model}_split_mode_o_split_idx_0_.yml \
    --split_idx $split_idx \
    --split_mode $split_mode \
    --name $checkpoint_name
done
