DATASET=$1
MODELS=$2

# define GPU and CPU for srun
GPU=1
CPU=16


echo "TESTS with $GPU GPUs and $CPU CPUs with [ $DATASET | $MODELS ]"

# run the run.sh script
sbatch --gpus=$GPU --cpus-per-task=$CPU --job-name=$DATASET -o logs/single_test/$DATASET --wrap="./src/.run_scripts/run.sh test_transforms.py $DATASET $MODELS"
