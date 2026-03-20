#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adversarial_evaluation.out

unset SLURM_EXPORT_ENV
module load python

# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .


# Set up directories
export PROJECT_DIR="$HOME/rosaid"
INPUT_ROOT="/rosaid_targeted/results/adversarial_attacks/ROSIDS23_blockcnn2d__nosampling_binary_20260102"
OUTPUT_ROOT="$TMPDIR"
DATA_SPLIT="val"

# Extract last folder name from INPUT_ROOT for ADV_GEN_MODEL
ADV_GEN_MODEL=$(basename "$INPUT_ROOT")

# Create adv_samples directory as OUTPUT_ROOT/ADV_GEN_MODEL
ADV_SAMPLES_DIR="$OUTPUT_ROOT/$ADV_GEN_MODEL"
mkdir -p "$ADV_SAMPLES_DIR"

echo "INPUT_ROOT:   $INPUT_ROOT"
echo "OUTPUT_ROOT:  $OUTPUT_ROOT"
echo "ADV_GEN_MODEL: $ADV_GEN_MODEL"
echo "ADV_SAMPLES_DIR: $ADV_SAMPLES_DIR"

echo "Processing adversarial samples for model: $ADV_GEN_MODEL"
echo "Data split to process: $DATA_SPLIT"
echo "Input directory: $INPUT_ROOT"
echo "Output directory: $ADV_SAMPLES_DIR"
echo ""

# Determine which splits to process
if [ "$DATA_SPLIT" == "both" ]; then
    splits_to_process=("train" "val")
elif [ "$DATA_SPLIT" == "train" ]; then
    splits_to_process=("train")
elif [ "$DATA_SPLIT" == "val" ]; then
    splits_to_process=("val")
else
    echo "Error: DATA_SPLIT must be 'train', 'val', or 'both'"

fi

# Process each attack method
for method_dir in "$INPUT_ROOT"/*; do
    if [ -d "$method_dir" ]; then
        method_name=$(basename "$method_dir")
        echo "Processing method: $method_name"
        
        # Unzip specified split files
        for split in "${splits_to_process[@]}"; do
            # Create output directory with train/val subdirectory
            output_method_dir="$ADV_SAMPLES_DIR/$ADV_GEN_MODEL/$method_name/$split"
            mkdir -p "$output_method_dir"
            
            zip_file="$method_dir/${method_name}_${split}.zip"
            if [ -f "$zip_file" ]; then
                echo "  - Unzipping ${method_name}_${split}.zip to $output_method_dir"
                unzip -q "$zip_file" -d "$output_method_dir"
                echo "    ✓ Extracted ${split} data"
            else
                echo "  - Warning: $zip_file not found"
            fi
        done
        echo ""
    fi
done

echo "All adversarial samples processed!"
echo "Output location: $ADV_SAMPLES_DIR/$ADV_GEN_MODEL"

# # first rosids stfpm 
python rosaid/evaluation/adversarial_evaluation.py \
    --data_dir $ADV_SAMPLES_DIR/$ADV_GEN_MODEL --batch_size 32 \
    --clf_model_path rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
    --stfpm_model_path rosaid/results/image_stfpm/blockcnn2d___ROSIDS23_nosampling_20260108 \
    --max_data -1 --project_dir $PROJECT_DIR

# # now use stfpm based on dnp3
python rosaid/evaluation/adversarial_evaluation.py \
    --data_dir $ADV_SAMPLES_DIR/$ADV_GEN_MODEL --batch_size 32 \
    --clf_model_path rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
    --stfpm_model_path rosaid/results/image_stfpm/DNP3_blockcnn2d___ROSIDS23_nosampling_20260108 \
    --max_data -1 --project_dir $PROJECT_DIR

# python rosaid/evaluation/adversarial_evaluation.py \
#     --data_dir $ADV_SAMPLES_DIR/$ADV_GEN_MODEL --batch_size 32 \
#     --clf_model_path rosaid/results/image_classification/ROSIDS23_blockcnn2d__nosampling_binary_20260111 \
#     --stfpm_model_path rosaid/results/image_stfpm/results_resnet18___ROSIDS23_nosampling_20260124 \
#     --max_data -1 --project_dir $PROJECT_DIR