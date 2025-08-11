#!/bin/bash

#SBATCH -p gpu                 # GPU partition
#SBATCH -N 1                   # One node
#SBATCH --mem=40G              # 40 GB RAM
#SBATCH --gpus-per-node=1      # One GPU
#SBATCH -n 30                  # 30 CPU cores
#SBATCH -t 5-00:00:00          # 5 days
#SBATCH --mail-type=END,FAIL   # Notify on end or fail
#SBATCH --job-name=spike_sorting
#SBATCH -o spike_sorting_%j.out
#SBATCH -e spike_sorting_%j.err

# -------------------------------
# LOAD MODULES
# -------------------------------
bash
module load mamba
module load cuda/12.8   # <-- match to your cluster’s available CUDA module

#eval $(micromamba shell hook bash)"
mamba activate si_ks4_env

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
