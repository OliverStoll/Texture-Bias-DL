# define GPU and CPU for srun
GPU=1
CPU=8

# take hours as input
hours=$1
# zero pad the hours if less than 2 digits
if [ ${#hours} -eq 1 ]; then
    hours=0$hours
fi


TIMESTAMP="$hours:00:00"
echo "Running for $TIMESTAMP with $GPU GPUs and $CPU CPUs"

# run the run.sh script
srun --gpus=$GPU --cpus-per-task=$CPU --time=$TIMESTAMP ./src/.run_scripts/run.sh