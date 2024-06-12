#!/usr/bin/env bash

pip3 install torch torchvision scikit-learn
pip3 install transformers datasets evaluate accelerate tensorboard

mkdir -p /scratch/experiment
cp /shared/data/dibas/UMMDS.zip /scratch/experiment
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

unzip -q /shared/data/dibas/UMMDS.zip -d /scratch/experiment/ummds
cd dibas_ummds
python3 experiment.py --epoch=10 --model_name_or_path='microsoft/resnet-18' --data_dir='/scratch/experiment/ummds/UMMDS' --work_dir='/scratch/experiment/work'

cp -r /scratch/experiment/work/ /shared/data/dibas

# copy trained model and logs to /shared/data