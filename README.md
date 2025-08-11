# Spike Sorting Pipeline

Internal Akrami Lab pipeline for spike sorting Neuropixels and other high-density recordings using [SpikeInterface](https://spikeinterface.readthedocs.io) and **Kilosort 4**.  
Supports **Open Ephys** and **SpikeGLX** data formats.  

This repository contains:

- An **interactive script** for local step-by-step testing and development in VS Code.
- A **batch (SLURM) script** for large-scale spike sorting jobs on HPC systems.
- Environment setup for GPU-accelerated sorting with Kilosort 4.



## 1. Environment Setup (Interactive Use)

The interactive pipeline is intended for **local testing or prototyping** before running large jobs on the cluster.

### 1.1 Create the Conda Environment

```bash
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env
```

Environment contents (from `si_ks4_env.yml`):

- Python 3.9
- SpikeInterface (core, extractors, preprocessors, sorters, qualitymetrics)
- probeinterface
- matplotlib, numpy
- CUDA toolkit for GPU-based sorting
- Other dependencies for Open Ephys and SpikeGLX reading

### 1.2 Running the Interactive Script

There are two options:

#### Option A – **Line-by-line in VS Code**

1. Open the repository in VS Code.
2. Select the `si_ks4_env` Python interpreter.
3. Open the interactive script (`spikeinterface_script_interactive.py`).
4. Run **cell by cell** (`# %%` sections) to inspect intermediate steps.

#### Option B – **Full run from terminal**

```bash
micromamba activate si_ks4_env
python spikeinterface_script_interactive.py
```

You will be prompted (or you can edit the script) to set:

- `data_path` → location of raw recording (Open Ephys or SpikeGLX folder)
- `output_path` → where results will be saved



## 2. Batch Processing on HPC (SLURM)

For large datasets, use the batch script to submit jobs to the GPU queue.

### 2.1 No Conda Environment Needed

On the cluster, the script uses **pre-installed modules**.
Check `run_spikeinterface.sh` in the repo for exact module loading lines.

Typical workflow:

```bash
sbatch run_spikeinterface.sh \
    /path/to/rawdata/.../recording1 \
    /path/to/output_dir \
    --show_preprocessing
```

Where:

- First argument: input recording folder
- Second argument: output folder
- Optional flags: e.g. `--show_preprocessing` to save preprocessed plots



## 3. VS Code Remote Development with HPC

If you want to run the interactive script **directly on the HPC** (instead of locally):

1. Install the **Remote-SSH** extension in VS Code.
2. Add your HPC login to `~/.ssh/config`:

   ```ssh
   Host swc-hpc
       HostName gpu-login.yourcluster.ac.uk
       User yourusername
       IdentityFile ~/.ssh/id_rsa
   ```
3. In VS Code:

   * Press `F1` → "Remote-SSH: Connect to Host..."
   * Choose `swc-hpc`
4. Once connected, open this repository folder from the HPC filesystem.
5. Select the cluster's Python interpreter (with SpikeInterface/Kilosort support).
6. Run the interactive script as normal.



## 4. Kilosort GUI Not Required

Kilosort 4 runs entirely headless via SpikeInterface — **no MATLAB GUI needed**.



## 5. Viewing Results

Sorting outputs are stored in the `output_path` directory.

Inside you will find:

- `sorting/` – raw Kilosort output
- `waveforms/` – extracted unit waveforms
- `quality_metrics.csv` – automated metrics for all sorted units

You can inspect results using:

```python
from spikeinterface import load_extractor
from spikeinterface.widgets import plot_unit_waveforms

sorting = load_extractor("/path/to/output/sorting")
plot_unit_waveforms(sorting)
```

For a full analysis interface, you can also load results into:

- **Phy** (SpikeInterface can export)
- Or use SpikeInterface widgets for visualization



## 6. Data Notes

- Open Ephys → TTLs are in a separate events file.
- SpikeGLX → TTLs are stored as an extra channel.
- Sorting pipeline works identically for both — only the loading function changes.

