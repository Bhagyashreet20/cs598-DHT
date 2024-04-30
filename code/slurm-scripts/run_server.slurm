#!/bin/bash

#SBATCH --mem=32g
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcng-delta-gpu
#SBATCH --job-name=start_x_server 
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00 
#SBATCH --constraint="scratch"
#SBATCH --output=/projects/bcng/cs598-DHT/code/logs/slurm-logs/output/x_server_start_-%j.out  # Standard output log
#SBATCH --error=/projects/bcng/cs598-DHT/code/logs/slurm-logs/error/x_server_start_-%j.err   # Standard error log

# Load necessary modules
module load gcc python/3.8.18
module load anaconda3_gpu
module load cuda
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

# export DISPLAY=:0
# # export XAUTHORITY=/tmp/.Xauthority_$SLURM_JOB_ID
# mkdir -p /tmp/.X11-unix
# chmod 1777 /tmp/.X11-unix

# export DISPLAY=:0
# export XAUTHORITY=$HOME/.Xauthority

# Start Xvfb
# Xvfb $DISPLAY -screen 0 1024x768x24 -auth $XAUTHORITY &

# Starting Xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

echo "Xvfb started on display ${DISPLAY}"
sleep 5  # Give some time for Xvfb to start

echo "Starting remote server"
# Run the training script
# Running the script with X11 forwarding
srun --x11=first,local python /projects/bcng/ukakarla/teach/bin/startx.py
