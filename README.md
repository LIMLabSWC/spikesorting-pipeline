# **Spike Sorting Pipeline – Akrami Lab**

Pipeline for spike sorting **Neuropixels** and other high-density recordings using [SpikeInterface](https://spikeinterface.readthedocs.io) and **Kilosort 4**.
Supports **Open Ephys** and **SpikeGLX** formats.


## **Contents**

1. Overview of repository files
2. Environment setup (GPU-ready)
3. Interactive HPC development
4. Local development (optional)
5. Batch processing on HPC
6. Results viewing and analysis
7. Data format details
8. Pipeline features
9. Known issues
10. Troubleshooting


## **1. Repository Contents**

This repository includes:

| File                                   | Purpose                                                      |
| -------------------------------------- | ------------------------------------------------------------ |
| `spikeinterface_script_interactive.py` | Step-by-step testing and development in VS Code              |
| `spikeinterface_batch.py`              | Large-scale spike sorting jobs on HPC                        |
| `view_sorting_results.py`              | Analyze and visualize sorting results                        |
| `si_ks4_env.yml`                       | Environment file for GPU-accelerated sorting with Kilosort 4 |


## **2. Environment Setup**

**Goal:** Install all dependencies for SpikeInterface + Kilosort 4, ready for GPU use.

### **2.1 Create the Conda Environment**

```bash
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env
```

**Included packages:**

| Package                            | Purpose                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| Python 3.11                        | Base language                                                                       |
| SpikeInterface 0.103.0             | Core spike sorting tools (core, extractors, preprocessors, sorters, qualitymetrics) |
| Kilosort 4.1.0                     | GPU-accelerated sorter                                                              |
| probeinterface                     | Probe geometry handling                                                             |
| matplotlib, numpy                  | Plotting and analysis                                                               |
| CUDA toolkit                       | GPU sorting                                                                         |
| Open Ephys & SpikeGLX dependencies | Data loading                                                                        |


## **3. Interactive Development on HPC**

**Goal:** Use the interactive pipeline for step-by-step GPU-accelerated spike sorting. Recommended for testing and debugging.


### **3.1 HPC Setup**

1. **Install Remote-SSH extension in VS Code**

   * Extensions → Search "Remote - SSH" → Install.

2. **Configure SSH access**
   Add to `~/.ssh/config`:

   ```ssh
   Host bastion
       Hostname ssh.swc.ucl.ac.uk
       User vplattner

   Host hpc2
       Hostname hpc-gw2
       User vplattner
       ProxyJump bastion
   ```

5. **Clone the repository**

   ```bash
   git clone git@github.com:LIMLabSWC/spikesorting-pipeline.git
   cd spikesorting-pipeline
   ```

4. **Install environment on HPC**

   ```bash
   micromamba create -n si_ks4_env -f si_ks4_env.yml
   ```

5. **Request a GPU compute node**
   **For testing**:

   ```bash
   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=50G --pty bash -i
   ```

6. **Activate environment inside compute node**

   ```bash
   micromamba activate si_ks4_env
   ```


### **3.2 Running the Interactive Script**

1. **Connect to HPC via VS Code**

   * Press `F1` → “Remote-SSH: Connect to Host…” → Select `hpc2`
   * Enter password → Wait for connection

2. **Open repository folder on HPC**
   Path: `/nfs/nhome/live/vplattner/spikesorting-pipeline`

3. **Select Python interpreter**
   `Ctrl+Shift+P` → “Python: Select Interpreter” →
   `/nfs/nhome/live/vplattner/micromamba/envs/si_ks4_env/bin/python`

4. **Run the script interactively**

   * Open `spikeinterface_script_interactive.py`
   * Run cells with `Shift+Enter`
   * Runs on GPU inside compute node

**Note:** Keep your compute node session active. If it expires, request a new node and reconnect.


## **4. Local Development (Optional)**

**Goal:** Run the interactive pipeline on your local machine (requires GPU + CUDA).

```bash
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env
python spikeinterface_script_interactive.py
```

**Note:** Modify hardcoded paths in the script for your dataset.


## **5. Batch Processing on HPC (SLURM)** - *(Not fully tested)*

**Goal:** Submit large dataset jobs to the GPU queue. 

* Uses pre-installed HPC modules (see `run_spikeinterface.sh`).
* No local Conda env needed.

**Example:**

```bash
sbatch run_spikeinterface.sh \
    /path/to/rawdata/.../recording1 \
    /path/to/output_dir \
    --show_preprocessing
```

**Arguments:**

1. Input recording folder
2. Output folder
3. *(Optional)* flags, e.g. `--show_preprocessing`


## **6. Results Viewing and Analysis**

**Goal:** Inspect sorting results with summary statistics and plots.

```bash
python view_sorting_results.py /path/to/sorting/results
```

**Features:**

* Summary statistics (units, spikes, firing rates)
* Quality metrics (SNR, ISI violations, presence ratio)
* Plots:

  * Waveforms
  * Raster plots
  * Metrics distributions
  * Unit locations
  * Autocorrelograms

**Output structure:**

```
output_path/
├── sorting/
├── postprocessing/
│   └── quality_metrics.csv
└── plots/
    ├── unit_waveforms.png
    ├── raster_plot.png
    ├── quality_metrics_distributions.png
    ├── unit_locations.png
    └── autocorrelograms.png
```

## **7. Data Format Support**

**Open Ephys:**

* TTLs in separate events file
* Geometry from `settings.xml`

**SpikeGLX:**

* TTLs as extra channel
* Geometry manual or from metadata


## **8. Pipeline Features**

* **Preprocessing:**

  1. Phase shift correction
  2. Bandpass filter (300–6000 Hz)
  3. Common reference
  4. Probe geometry attachment

* **Sorting:**

  * Kilosort 4 with GPU acceleration
  * Single-shank grouping
  * Sparse waveform extraction (75 µm radius)

* **Metrics:**

  * SNR
  * ISI violations
  * Presence ratio
  * Firing rate
  * Spike count

## **9. Known Issues**

* Hardcoded paths in interactive script
* Quality metrics unreliable for < 60 s recordings
* CUDA version must match cluster (currently 12.2)
* Large files may require 40 GB+ RAM


## **10. Troubleshooting**

* **Missing settings.xml:** Needed for probe geometry
* **CUDA errors:** Check GPU and CUDA version
* **Memory errors:** Increase SLURM memory or shorten data
* **Metric warnings:** Normal for short recordings

