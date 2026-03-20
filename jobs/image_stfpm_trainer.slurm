#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=image_stfpm_trainer.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

# IEC104, DNP3, ROSIDS23
export DATA_TYPE="ROSIDS23"
# blockcnn2d, resnet18
export BACKBONE="resnet18"

export IEC104_TEACHER_NORMAL="rosaid/results/image_classification/IEC104_blockcnn2d__nosampling_multiclass_20251231/best_model_full.pth"
export IEC104_TEACHER_NORMALIZED="rosaid/results/image_classification/IEC104_blockcnn2d__normalized_nosampling_multiclass_20251231/best_model_full.pth"
export DNP3_TEACHER_NORMAL="rosaid/results/image_classification/DNP3_blockcnn2d__nosampling_multiclass_20251231/best_model_full.pth"
export DNP3_TEACHER_NORMALIZED="rosaid/results/image_classification/DNP3_blockcnn2d__normalized_nosampling_multiclass_20251230/best_model_full.pth"
export ROSIDS23_TEACHER_NORMAL="rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260102/best_model_full.pth"
export ROSIDS23_TEACHER_NORMALIZED=""
# for now use dnp3 model
# export ROSIDS23_TEACHER_NORMAL="rosaid/results/image_classification/DNP3_blockcnn2d__nosampling_binary_20260107/best_model_full.pth"
export ROSIDS23_TEACHER_NORMALIZED=""


if [[ "$DATA_TYPE" == "IEC104" ]]; then
    export DATA_DIR="$HOME/rosaid/data/iec104"
    export TEACHER_MODEL_NORMAL_PATH="$IEC104_TEACHER_NORMAL"
    export TEACHER_MODEL_NORMALIZED_PATH="$IEC104_TEACHER_NORMALIZED"
elif [[ "$DATA_TYPE" == "DNP3" ]]; then
    export DATA_DIR="$HOME/rosaid/data/dnp3"
    export TEACHER_MODEL_NORMAL_PATH="$DNP3_TEACHER_NORMAL"
    export TEACHER_MODEL_NORMALIZED_PATH="$DNP3_TEACHER_NORMALIZED"
elif [[ "$DATA_TYPE" == "ROSIDS23" ]]; then
    export DATA_DIR="$HOME/rosaid/data/rosids23"
    export TEACHER_MODEL_NORMAL_PATH="$ROSIDS23_TEACHER_NORMAL"
    export TEACHER_MODEL_NORMALIZED_PATH="$ROSIDS23_TEACHER_NORMALIZED"
else
    echo "ERROR: Unknown DATA_TYPE='$DATA_TYPE'" >&2
    exit 1
fi

unzip -q "$DATA_DIR/session_images.zip" -d "$TMPDIR"
echo "Unzipped session_images.zip"
cp $DATA_DIR/labelled_sessions.csv $TMPDIR

export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/rosaid"
export SAMPLING_METHOD="nosampling"
export CLF_MODE="multiclass"

if [ "$BACKBONE" == "resnet18" ]; then
    export TEACHER_MODEL_NORMALIZED_PATH="$HOME/rosaid/results/resnet18_teacher.pth"
    export TEACHER_MODEL_NORMAL_PATH="$HOME/rosaid/results/resnet18_teacher.pth"
fi

python3 $HOME/rosaid/trainers/session_image_stfpm_trainer.py \
    --data_type $DATA_TYPE \
    --clf_mode $CLF_MODE --data_dir $DATA_DIR \
    --project_dir $PROJECT_DIR --image_type normal \
    --batch_size 64 \
    --teacher_model $TEACHER_MODEL_NORMAL_PATH \
    --backbone $BACKBONE \
    --max_data -1


# python3 $HOME/rosaid/trainers/session_image_stfpm_trainer.py \
#     --data_type $DATA_TYPE \
#     --clf_mode $CLF_MODE --data_dir $DATA_DIR \
#     --project_dir $PROJECT_DIR --image_type normalized \
#     --batch_size 64 \
#     --teacher_model $TEACHER_MODEL_NORMALIZED_PATH \
#     --backbone $BACKBONE \
#     --max_data -1



