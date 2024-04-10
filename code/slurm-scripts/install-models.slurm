#!/bin/bash

#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --account=bcng-delta-cpu
#SBATCH --job-name=t5-base
#SBATCH --time=00:15:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/t5-small.%j.out
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/t5-small.%j.err

#by default the models will be downloaded to /u/btaleka{user acct running script}/.cache/huggingface/hub so changing the cache folder to scratch. 
export TRANSFORMERS_CACHE=/scratch/bcng/cs598-DHT/models


module load gcc python
module load anaconda3_cpu
module list

conda activate cs598-DHT
pip install -r /projects/bcng/cs598-DHT/code/python/requirements.txt
pip install --upgrade transformers

echo "job is starting on `hostname`"
srun python3 /projects/bcng/cs598-DHT/code/python/t5-base.py #- > --mem=16g
#srun python3 /projects/bcng/cs598-DHT/code/python/t5-xxl.py -> -mem=64g and change time too time=03:00:00
#srun python3 /projects/bcng/cs598-DHT/code/python/mixtral-8x7B.py  #--mem=128g , --time=03:00:00
