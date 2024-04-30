#!/bin/bash

#SBATCH --mem=32g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcng-delta-gpu
#SBATCH --job-name=lmdb_creation
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1  
#SBATCH --time=12:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/lmdb_creation-%j.out  # Standard output log
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/lmdb_creation-%j.err   # Standard error log

#by default the models will be downloaded to /u/btaleka{user acct running script}/.cache/huggingface/hub so changing the cache folder to scratch. 
module load gcc python/3.8.18
module load anaconda3_gpu
module list

conda activate teachllms
conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r /projects/bcng/ukakarla/teach/requirements.txt

export ET_DATA='/projects/bcng/cs598-DHT/teach_data'
export TEACH_ROOT_DIR='/projects/bcng/ukakarla/teach'
export ET_LOGS='/projects/bcng/cs598-DHT/teach_data/et_pretrained_models'
export TEACH_SRC_DIR="$TEACH_ROOT_DIR/src"
export ET_ROOT="$TEACH_SRC_DIR/teach/modeling/ET"
export INFERENCE_OUTPUT_PATH='/projects/bcng/ukakarla/teach/inference_output'
export PYTHONPATH="$TEACH_SRC_DIR:$ET_ROOT:$PYTHONPATH"

echo "Environment variables set:"
echo "PYTHONPATH=$PYTHONPATH"
echo "ET_DATA=$ET_DATA"
echo "ET_LOGS=$ET_LOGS"
echo "TEACH_ROOT_DIR=$TEACH_ROOT_DIR"
echo "TEACH_SRC_DIR=$TEACH_SRC_DIR"
echo "ET_ROOT=$ET_ROOT"

echo "Starting LMDB creation on `hostname`"
# Running the script without needing to navigate to the folder
srun python -m alfred.data.create_lmdb with args.visual_checkpoint="$ET_LOGS/pretrained/fasterrcnn_model.pth" \
    args.data_input=edh_instances \
    args.task_type=edh \
    args.data_output=lmdb_edh \
    args.vocab_path=None
