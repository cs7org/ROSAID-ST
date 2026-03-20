#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=image_trainer_binary.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

# IEC104, DNP3, ROSIDS23
export DATA_TYPE="ROSIDS23"
# In paper: ORIGINAL, NORMALIZED
export NORMAL_TYPES=("ORIGINAL" "FILTERED" "NORMALIZED" "FILTERED_GRAM" "NORMALIZED_GRAM" "ZSCORE" "ZGRAM1D" "ZGRAM3D" "UNFILTERED_GRAM" "UNFILTERED_GRAM3D" "FILTERED_GRAM3D")
export NORMAL_TYPES=("ORIGINAL" "NORMALIZED")

if [[ "$DATA_TYPE" == "IEC104" ]]; then
    export DATA_DIR="$HOME/rosaid/data/iec104"
    export BATCH_SIZE=64
elif [[ "$DATA_TYPE" == "DNP3" ]]; then
    export DATA_DIR="$HOME/rosaid/data/dnp3"
    export BATCH_SIZE=128
elif [[ "$DATA_TYPE" == "ROSIDS23" ]]; then
    export DATA_DIR="$HOME/rosaid/data/rosids23"
    export BATCH_SIZE=40
else
    echo "ERROR: Unknown DATA_TYPE='$DATA_TYPE'" >&2
    exit 1
fi

echo "Unzipping files to $TMPDIR"
unzip -q "$DATA_DIR/session_images.zip" -d "$TMPDIR"
echo "Unzipped session_images.zip"
cp $DATA_DIR/labelled_sessions.csv $TMPDIR

export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/rosaid"
export SAMPLING_METHOD="nosampling"
# binary,multiclass
export CLF_MODE="binary"
# "mobilenet_v3_large, resnet18, blockcnn2d"
export BACKBONE="blockcnn2d"
export NORMAL_TYPE="ORIGINAL"

for NORMAL_TYPE in "${NORMAL_TYPES[@]}"; do
    echo "Training with NORMAL_TYPE=$NORMAL_TYPE"

    python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
        --data_type $DATA_TYPE --clf_mode $CLF_MODE \
        --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
        --backbone "blockcnn2d" --image_type normal \
        --batch_size 32 \
        --max_data -1 --normal_image_type $NORMAL_TYPE &
    python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
        --data_type $DATA_TYPE --clf_mode $CLF_MODE \
        --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
        --backbone "mobilenet_v3_large" --image_type normal \
        --batch_size 32 \
        --max_data -1 --normal_image_type $NORMAL_TYPE &
    python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
        --data_type $DATA_TYPE --clf_mode $CLF_MODE \
        --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
        --backbone "resnet18" --image_type normal \
        --batch_size 32 \
        --max_data -1 --normal_image_type $NORMAL_TYPE
    # wait for it to finish
    wait
    

done

# python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
#     --data_type $DATA_TYPE --clf_mode $CLF_MODE \
#     --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
#     --backbone "blockcnn2d" --image_type normalized \
#     --batch_size 32 \
#     --max_data -1 --normal_image_type $NORMAL_TYPE &
# python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
#     --data_type $DATA_TYPE --clf_mode $CLF_MODE \
#     --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
#     --backbone "mobilenet_v3_large" --image_type normalized \
#     --batch_size 32 \
#     --max_data -1 --normal_image_type $NORMAL_TYPE &
# python3 $HOME/rosaid/trainers/session_image_clf_trainer.py \
#     --data_type $DATA_TYPE --clf_mode $CLF_MODE \
#     --data_dir $DATA_DIR --project_dir $PROJECT_DIR \
#     --backbone "resnet18" --image_type normalized \
#     --batch_size 32 \
#     --max_data -1 --normal_image_type $NORMAL_TYPE