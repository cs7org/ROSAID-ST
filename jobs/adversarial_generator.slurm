#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adversarial_generator.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

echo "Unzipping files to $TMPDIR"
unzip -q "$HOME/rosaid/data/rosids23/session_images.zip" -d "$TMPDIR"
echo "Unzipped session_images.zip"
cp $HOME/rosaid/data/rosids23/labelled_sessions.csv $TMPDIR

export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/rosaid"
export SAVE_DIR="$WORK/rosaid"
# make sure the save directory exists else change it
# dont create if already exists
if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
fi
echo 'Generating adversarial images to '$SAVE_DIR

python3 $HOME/rosaid/data_preparation/generate_adversarial_image.py \
    --data_dir $DATA_DIR --project_dir $PROJECT_DIR --save_dir $SAVE_DIR \
    --model_names ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
    --image_type normal --batch_size 64 --max_data -100 --validation_only

# python3 $HOME/rosaid/data_preparation/generate_adversarial_image.py \
#     --data_dir $DATA_DIR --project_dir $PROJECT_DIR --save_dir $SAVE_DIR \
#     --model_names ROSIDS23_blockcnn2d__normalized_nosampling_binary_20260106 \
#     --image_type normalized --batch_size 32 --max_data -100 --validation_only
