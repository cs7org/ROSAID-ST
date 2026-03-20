#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=rosids23_pcap_to_image.out

unset SLURM_EXPORT_ENV
module load python
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .
# NOTE: only install required packages instead of whole 
# pip install "scapy==2.6.1" loguru "opencv-python>=4.10" "tqdm>=4.62.3" "pydantic>=1.10.0" "pandas>=1.5.0"

export DATA_DIR="$HOME/rosaid/data/rosids23"
export OUT_DIR="$WORK/rosids23_session_images"
# unzip all .zip files inside datadir to $TMPDIR
for file in $DATA_DIR/*.zip; do
    unzip -q "$file" -d "$TMPDIR"
done

# move all unzipped pcap and csv files from sub folders in $TMPDIR to $TMPDIR
# and ignore error if no such files
mv $TMPDIR/**/*.pcap $TMPDIR 2>/dev/null
mv $TMPDIR/**/*.csv $TMPDIR 2>/dev/null

# now install apptainer
apptainer pull cincan_tshark.sif docker://cincan/tshark
# verify installation
apptainer exec --no-home --cleanenv cincan_tshark.sif tshark --version
apptainer exec --no-home --cleanenv cincan_tshark.sif editcap --version

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for Slurm version >22.05: cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

python3 $HOME/rosaid/data_preparation/rosids23_pcap_to_img_mp.py \
    --data_dir "$TMPDIR" \
    --project_dir "$HOME/rosaid" \
    --out_dir "$OUT_DIR" \
    --compress_to "$OUT_DIR/rosids23_session_images.zip" \
    --temp_dir "$TMPDIR" \
    --num_processes $SLURM_CPUS_PER_TASK \
    --use_apptainer \
    --container "$HOME/rosaid/cincan_tshark.sif"