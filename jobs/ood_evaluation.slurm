#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=ood_evaluation.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

# IEC104, DNP3, ROSIDS23
# unzip all 
export DATA_TYPES=("IEC104" "DNP3" "ROSIDS23")
for DATA_TYPE in "${DATA_TYPES[@]}"; do
    if [[ "$DATA_TYPE" == "IEC104" ]]; then
        export DATA_DIR="$HOME/rosaid/data/iec104"
    elif [[ "$DATA_TYPE" == "DNP3" ]]; then
        export DATA_DIR="$HOME/rosaid/data/dnp3"
    elif [[ "$DATA_TYPE" == "ROSIDS23" ]]; then
        export DATA_DIR="$HOME/rosaid/data/rosids23"
    else
        echo "ERROR: Unknown DATA_TYPE='$DATA_TYPE'" >&2
        exit 1
    fi
    # make sub dir
    mkdir -p "$TMPDIR/$DATA_TYPE"
    # unzip only if archive exists
    if [[ -f "$DATA_DIR/session_images.zip" ]]; then
        echo "Unzipping files to $TMPDIR/$DATA_TYPE"
        unzip -q "$DATA_DIR/session_images.zip" -d "$TMPDIR/$DATA_TYPE"
        echo "Unzipped session_images.zip"
        cp $DATA_DIR/labelled_sessions.csv $TMPDIR/$DATA_TYPE
    else
        echo "WARNING: Archive $DATA_DIR/session_images.zip not found"
    fi


done

# but set DATA_DIR to TMPDIR
export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/rosaid"


# first rosids stfpm 
python rosaid/evaluation/ood_evaluation.py \
    --data_dir $DATA_DIR --batch_size 32 \
    --clf_model_path rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
    --stfpm_model_path rosaid/results/image_stfpm/blockcnn2d___ROSIDS23_nosampling_20260108 \
    --max_data -1 --project_dir $PROJECT_DIR

# now use stfpm based on dnp3
python rosaid/evaluation/ood_evaluation.py \
    --data_dir $DATA_DIR --batch_size 32 \
    --clf_model_path rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
    --stfpm_model_path rosaid/results/image_stfpm/DNP3_blockcnn2d___ROSIDS23_nosampling_20260108 \
    --max_data -1 --project_dir $PROJECT_DIR
    

