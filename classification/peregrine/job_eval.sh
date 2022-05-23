#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name=lpt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail

module load Python
source /data/$USER/.envs/ltp/bin/activate
python evaluation.py