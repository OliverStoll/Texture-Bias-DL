MODELS=$1

# define GPU and CPU for srun
GPU=1
CPU=8

for DATASET in "${ALL_DATASETS[@]}"
do
  echo "$DATASET: "
  # run the run.sh script and get a return value
  retval=$(sbatch --gpus=$GPU --cpus-per-task=$CPU --job-name=$DATASET -o logs/slurm/$DATASET.out --wrap="./src/.run_scripts/run.sh test_transforms.py $DATASET $MODELS")
done