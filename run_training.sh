DATASET=$1
MODELS=$2
PRETRAINED=$3

# define GPU and CPU for srun
RUNNER_SCRIPT_PATH="./src/.run_scripts/run.sh"
PYTHON_SCRIPT_NAME="run_training.py"
OUTPUT_LOG="logs/slurm/$DATASET.out"
GPU=1
CPU=8
hours=99

if [ ${#hours} -eq 1 ]; then  # zero pad the hours if less than 2 digits
    hours=0$hours
fi

echo "TRAINING with $GPU GPUs and $CPU CPUs with [ $DATASET | $MODELS | $PRETRAINED ]"

OUTPUT_LOG="logs/slurm/$DATASET.out"

# if no dataset is provided, iterate over all




if [[ -n "${DATASET}" ]]; then
  # take the dataset as only element in list
  echo "Training only on $DATASET"
  ALL_DATASETS=($DATASET)
else
  echo "Training on all datasets"
  ALL_DATASETS=("imagenet" "rgb-bigearthnet" "bigearthnet" "deepglobe" "caltech" "caltech-ft" "caltech-120")




for dataset in "${ALL_DATASETS[@]}"; do
# run the run.sh script
  echo "Starting training on $dataset"
  sbatch --gpus=$GPU --cpus-per-task=$CPU --job-name=dataset -o $OUTPUT_LOG --wrap="$RUNNER_SCRIPT_PATH $PYTHON_SCRIPT_NAME $dataset $MODELS $PRETRAINED"
done
