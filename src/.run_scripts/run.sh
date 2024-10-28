#!/bin/bash

SCRIPT_NAME=$1
DATASET=$2
MODELS=$3

# Path to your virtual environment
VENV_PATH="/home/olivers/master-thesis/venv"
INCLUDE_PATH="/home/olivers/master-thesis/src"

export PYTHONPATH=$PYTHONPATH:$INCLUDE_PATH

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# RUN TRAINING WITH SRUN
python /home/olivers/master-thesis/src/.run_scripts/$SCRIPT_NAME $DATASET $MODELS

# Deactivate the virtual environment after the script finishes
deactivate
