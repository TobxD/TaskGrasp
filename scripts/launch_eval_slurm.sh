#!/bin/sh

if [[ $# -ne 3 ]]
then
  echo "usage: sh launch_train_slurm.sh <name_prefix> <model> <split_mode>"
  echo "example: sh launch_train_slurm.sh change_layer_num_42 sgn o"
  echo 'There have to be checkpoints with name prefix "<name_prefix>_<model>_<split_mode>_<split_idx>]"'
  echo "If there are multiple such folders, the one with the latest timestamp is taken"
  exit
fi

name_prefix=$1
model=$2
split_mode=$3

weight_files=()
for split_idx in 0 1 2 3; do
  prefix="${name_prefix}_${model}_${split_mode}_${split_idx}"
  files=($(find checkpoints -maxdepth 1 -name "${prefix}*" -printf "%f\n" | sort))

  # Check if there are any matching files
  if [ ${#files[@]} -eq 0 ]; then
      echo "No suitable experiment folder found with prefix ${prefix}"
      exit 1
  fi
  # More than one match -> we take the last one
  if [ ${#files[@]} -gt 1 ]; then
      echo "More than one experiment folder found with prefix ${prefix}"
      echo "Taking ${files[-1]} out of (${files[@]})"
      echo
  fi
  
  weight_files+=("${files[-1]}")
done

echo "continuing with these weight files for the 4 splits:"
echo "${weight_files[@]}"
echo

# for dataset in "train" "val" "test"; do
for dataset in "train" "val" "test"; do
  echo "launching for $dataset"
  all_job_ids=""
  for split_idx in 0 1 2 3; do
    job_id=$(sbatch --parsable eval_1gpu_12gb.sh \
      cfg/train/${model}/${model}_split_mode_o_split_idx_0_.yml \
      --save  \
      --dataset_name $dataset \
      --split_idx ${split_idx} \
      --split_mode $split_mode \
      --weight_file ${weight_files[split_idx]})
    echo "submitted $job_id for ${weight_files[split_idx]}"
    all_job_ids="$all_job_ids:$job_id"
  done

  # Do the plotting as soon as the evaluations are done
  echo "$all_job_ids as dependency for plotting"
  sbatch --dependency=afterok$all_job_ids plot.sh \
    --name_prefix $name_prefix \
    --dataset_name $dataset \
    --split_mode $split_mode \
    --model $model
done
