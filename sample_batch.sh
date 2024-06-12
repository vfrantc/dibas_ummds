#!/usr/bin/env bash
#SBATCH --partition=reu-gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=batch_job

# Constants
EPOCHS=10
MODEL_NAME_OR_PATH='google/efficientnet-b3'
DATA_DIR='/scratch/experiment/ummds/UMMDS'
WORK_DIR='/scratch/experiment/work'
GIT_REPO_URL='https://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@github.com/vfrantc/dibas_ummds.git'
LOCAL_REPO_DIR='/scratch/experiment/dibas_ummds'

# Load required modules
module load python/3.8

# Create necessary directories
mkdir -p /scratch/experiment

# Copy or clone the repository
if [ ! -d "$LOCAL_REPO_DIR" ]; then
  git clone $GIT_REPO_URL $LOCAL_REPO_DIR
fi

# Copy data
cp /shared/data/dibas/UMMDS.zip /scratch/experiment
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -q /scratch/experiment/UMMDS.zip -d /scratch/experiment/ummds

# Navigate to the repository
cd $LOCAL_REPO_DIR

# Install necessary packages
pip3 install --user torch torchvision scikit-learn transformers datasets evaluate accelerate tensorboard

# Run the experiment
python3 experiment.py --epoch=$EPOCHS --model_name_or_path=$MODEL_NAME_OR_PATH --data_dir=$DATA_DIR --work_dir=$WORK_DIR

# Copy the results back to shared data directory
cp -r $WORK_DIR /shared/data/dibas