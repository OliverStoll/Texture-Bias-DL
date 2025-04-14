#!/bin/bash

SCRIPT_NAME=$1
DATASET=$2
MODELS=$3
PRETRAINED=$4

# Path to your virtual environment
VENV_PATH="/home/olivers/master-thesis/venv"
INCLUDE_PATH="/home/olivers/master-thesis/src"

# if 'test' in script_name, set env-var DISABLE_ARTIFACTS to disable saving artifacts and echo this
if [[ $SCRIPT_NAME == *"test"* ]]; then
    export DISABLE_ARTIFACTS=1
    echo "DISABLED ARTIFACTS"
fi

export PYTHONPATH=$PYTHONPATH:$INCLUDE_PATH

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

echo "$all_packages"

# RUN TRAINING WITH SRUN
python /home/olivers/master-thesis/src/.run_scripts/$SCRIPT_NAME $DATASET $MODELS $PRETRAINED

# Deactivate the virtual environment after the script finishes
deactivate
