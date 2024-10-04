#!/bin/bash

# Path to your virtual environment
VENV_PATH="/home/olivers/master-thesis/venv"
INCLUDE_PATH="/home/olivers/master-thesis/src"

export PYTHONPATH=$PYTHONPATH:$INCLUDE_PATH

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# RUN TRAINING WITH SRUN
python /home/olivers/master-thesis/src/.run_scripts/run_training.py

# Deactivate the virtual environment after the script finishes
deactivate
