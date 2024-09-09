#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A40:1

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p opt_emb_configs.txt)
python opt_emb_train.py $args