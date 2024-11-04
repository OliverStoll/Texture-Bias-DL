DATASET=$1
MODELS=$2

# define GPU and CPU for srun
GPU=1
CPU=8
hours=99
# zero pad the hours if less than 2 digits
if [ ${#hours} -eq 1 ]; then
    hours=0$hours
fi


TIMESTAMP="$hours:00:00"
echo "TESTS ($TIMESTAMP) with $GPU GPUs and $CPU CPUs with [ $DATASET | $MODELS ]"

# run the run.sh script
sbatch --gpus=$GPU --cpus-per-task=$CPU --time=$TIMESTAMP --job-name=$DATASET -o logs/slurm/$DATASET.out --wrap="./src/.run_scripts/run.sh test_paired_transforms.py $DATASET $MODELS"
