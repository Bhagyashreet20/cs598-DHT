#!/bin/bash

#SBATCH --mem=128g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcng-delta-gpu
#SBATCH --job-name=mixtral-inference
#SBATCH --gpus=2        # Requesting a total of 3 GPUs
#SBATCH --ntasks=1              # Running 1 task
#SBATCH --gpus-per-node=2 # All 3 GPUs on the same node
#SBATCH --gpus-per-task=2    # All 3 GPUs for the single task
#SBATCH --time=00:30:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/mixtral-inference.%j.out
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/mixtral-inference.%j.err

#by default the models will be downloaded to /u/btaleka{user acct running script}/.cache/huggingface/hub so changing the cache folder to scratch.
#need to use minimum 3 gpus for mixtral as A100 has only 40GB memory
export HF_HOME=/scratch/bcng/cs598-DHT/models


module load gcc python
module load anaconda3_gpu
module list

conda activate cs598-DHT
pip install -r /projects/bcng/cs598-DHT/code/python/requirements.txt
pip install -U transformers
pip install -U accelerate


echo "job is starting on `hostname`"
srun python3 /projects/bcng/cs598-DHT/code/python/mixtral-8x7B.py  # , --time=03:00:00
#srun python3 /projects/bcng/cs598-DHT/code/python/t5-base.py