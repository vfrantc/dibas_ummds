#!/usr/bin/env bash

pip install torch torchvision scikit-learn
pip install transformers datasets evaluate accelerate

mkdir -p /scratch/experiment
cp /shared/data/dibas/UMMDS.zip /scratch/experiment
unzip -q /shared/data/dibas/UMMDS.zip -d /scratch/experiment/ummds

python3 experiment.py --network='microsoft/resnet-18'

# copy trained model and logs to /shared/data