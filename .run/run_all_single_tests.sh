MODELS=$1

# define GPU and CPU for srun
GPU=1
CPU=8

ALL_DATASETS=("imagenet" "bigearthnet" "rgb_bigearthnet" "caltech" "deepglobe" "caltech_120" "caltech_ft")

for DATASET in "${ALL_DATASETS[@]}"
do
  echo "$DATASET: "
  # run the run.sh script and get a return value
  retval=$(sbatch --gpus=$GPU --cpus-per-task=$CPU --job-name=$DATASET -o logs/single_test/$DATASET --wrap="./src/.run_scripts/run.sh test_transforms.py $DATASET $MODELS")
done