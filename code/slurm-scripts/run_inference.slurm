#!/bin/bash

#SBATCH --mem=128g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcng-delta-gpu
#SBATCH --job-name=et_model_inference
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/et_inference-%j.out  # Standard output log
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/et_inference-%j.err   # Standard error log

# Load necessary modules
module load gcc python/3.8.18
module load anaconda3_gpu
module list

# Activate your Python virtual environment
conda activate teachllms

# Ensure all necessary Python packages are installed
conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r /projects/bcng/ukakarla/teach/requirements.txt

# Set environment variables
export ET_DATA='/projects/bcng/ukakarla/teach_data'
export TEACH_ROOT_DIR='/projects/bcng/ukakarla/teach'
export ET_LOGS='/projects/bcng/ukakarla/teach_data/et_pretrained_models'
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

cd $TEACH_ROOT_DIR

echo "Running inference"
# Run the training script
srun python src/teach/cli/inference.py --model_module teach.inference.et_model \
    --model_class ETModel \
    --data_dir $ET_DATA \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_et_trial_latest \
    --split valid_seen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_et_trial_latest.json \
    --seed 4 \
    --model_dir $ET_LOGS/teach_et_trial_latest \
    --object_predictor $ET_LOGS/pretrained/maskrcnn_model.pth \
    --images_dir $ET_DATA/images \
    --visual_checkpoint $ET_LOGS/pretrained/fasterrcnn_model.pth
