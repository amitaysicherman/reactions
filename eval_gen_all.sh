#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-12
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p gen_configs.txt)
python eval_gen.py $args