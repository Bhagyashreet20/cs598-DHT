#!/bin/bash

#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --account=bcng-delta-cpu
#SBATCH --job-name=install-t5-base
#SBATCH --time=00:30:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/t5-base.%j.out
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/t5-base.%j.err

#by default the models will be downloaded to /u/btaleka{user acct running script}/.cache/huggingface/hub so changing the cache folder to scratch. 
export TRANSFORMERS_CACHE=/scratch/bcng/cs598-DHT/models


module load gcc python
module load anaconda3_cpu
module list

conda activate cs598-DHT
pip install -r /projects/bcng/cs598-DHT/code/python/requirements.txt

echo "job is starting on `hostname`"
srun python3 /projects/bcng/cs598-DHT/code/python/t5-base.py