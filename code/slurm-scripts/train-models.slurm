#!/bin/bash

#SBATCH --mem=32g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcng-delta-gpu
#SBATCH --job-name=train
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1  
#SBATCH --time=06:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/train.%j.out
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/train.%j.err

#by default the models will be downloaded to /u/btaleka{user acct running script}/.cache/huggingface/hub so changing the cache folder to scratch. 
export HF_HOME=/scratch/bcng/cs598-DHT


module load gcc python
module load anaconda3_gpu
module list

conda activate cs598-DHT
pip install -r /projects/bcng/cs598-DHT/code/python/requirements.txt
pip install -U transformers
pip install -U accelerate

echo "job is starting on `hostname`"
srun python3 /projects/bcng/cs598-DHT/code/python/train.py  #--mem=128g , --time=03:00:00
