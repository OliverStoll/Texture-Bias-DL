
# define GPU and CPU for srun
GPU=1
CPU=8

echo "TESTS ALL PAIRS with $GPU GPUs and $CPU CPUs"

# loop over all datasets
ALL_DATASETS=("imagenet" "bigearthnet" "rgb_bigearthnet" "caltech" "deepglobe" "caltech_120" "caltech_ft")

for DATASET in "${ALL_DATASETS[@]}"
do
  echo "Queuing sbatch job for $DATASET"
  # run the run.sh script
  sbatch --gpus=$GPU --cpus-per-task=$CPU --job-name=$DATASET -o logs/slurm/$DATASET.out --wrap="./src/.run_scripts/run.sh test_paired_transforms.py $DATASET"
done