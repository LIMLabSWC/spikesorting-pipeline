#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH -n 30
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=spike_sorting
#SBATCH -o spike_sorting_%j.out
#SBATCH -e spike_sorting_%j.err

# -------------------------------
# LOAD MODULES
# -------------------------------
module load mamba
module load cuda/12.2

# -------------------------------
# ACTIVATE MICROMAMBA ENV
# -------------------------------
eval "$(/nfs/nhome/live/vplattner/.local/bin/micromamba shell hook bash)"
micromamba activate si_ks4_env

# -------------------------------
# ARGUMENT HANDLING
# -------------------------------
DATA_PATH="$1"
OUTPUT_PATH="$2"

if [ -z "$DATA_PATH" ]; then
    echo "Usage: sbatch run_spikeinterface.sh <data_path> [output_path] [--show_probe] [--show_preprocessing]"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="./spike_sorting_output"
fi

shift 2 || true

# -------------------------------
# RUN SCRIPT
# -------------------------------
echo "Starting spike sorting with Kilosort 4..."
python ~/spikeinterface_script.py "$DATA_PATH" --output_path "$OUTPUT_PATH" "$@"
