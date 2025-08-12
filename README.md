# Spike Sorting Pipeline

Internal Akrami Lab pipeline for spike sorting Neuropixels and other high-density recordings using [SpikeInterface](https://spikeinterface.readthedocs.io) and **Kilosort 4**.

## Quick Start

### 1. Setup Environment
```bash
# On HPC
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env
```

### 2. Run Interactive Pipeline
```bash
# Get a compute node
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=50G --pty bash -i

# Run the script
python spikeinterface_script_interactive.py
```

### 3. View Results
```bash
python view_sorting_results.py /path/to/sorting/results
```

### 4. Manual Curation (Optional)
```bash
# Install Phy
conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
conda activate phy2
pip install git+https://github.com/cortex-lab/phy.git

# Launch GUI
cd /path/to/sorting/results
phy template-gui params.py
```

## What's Included

- **Interactive script** (`spikeinterface_script_interactive.py`) for step-by-step testing
- **Batch processing** (`spikeinterface_batch.py`) for large datasets
- **Results viewer** (`view_sorting_results.py`) for analysis and plots
- **Environment file** (`si_ks4_env.yml`) with all dependencies
- **SLURM script** (`run_spikeinterface.sh`) for HPC submission

## Output Structure

```
output_path/
├── sorting/           # Kilosort 4 output
├── postprocessing/    # Waveforms and quality metrics
│   └── quality_metrics.csv
└── plots/            # Visualization plots
    ├── unit_waveforms.png
    ├── raster_plot.png
    └── ...
```

---

## Detailed Documentation

### Environment Setup

The pipeline requires a Python environment with SpikeInterface, Kilosort 4, and related dependencies.

**Installation:**
```bash
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env
```

**Key dependencies:**
- SpikeInterface (latest)
- Kilosort 4 (via SpikeInterface)
- CUDA toolkit (for GPU acceleration)
- Other dependencies for Open Ephys and SpikeGLX reading

### Interactive Development on HPC

The interactive pipeline is designed for **step-by-step testing and debugging** 
of spike sorting workflows. **Recommended approach**: Run on HPC for GPU 
acceleration and better performance.

#### Setup for HPC Development

1. **Install the Remote-SSH extension** in VS Code.

2. **Configure SSH access** by adding your HPC login to `~/.ssh/config`:

   ```ssh
   Host bastion
       Hostname ssh.swc.ucl.ac.uk
       User vplattner

   Host hpc2
       Hostname hpc-gw2
       User vplattner
       ProxyJump bastion
   ```

3. **Install the environment** on HPC:
   ```bash
   micromamba create -n si_ks4_env -f si_ks4_env.yml
   ```

4. **Clone the repository**:
   ```bash
   git clone git@github.com:LIMLabSWC/spikesorting-pipeline.git
   cd spikesorting-pipeline
   ```

5. **Request a compute node** for GPU-accelerated processing:
   ```bash
   # Typical command for testing:
   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=50G --pty bash -i
   
   # For production runs, adjust resources as needed:
   # srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=100G --pty bash -i
   ```

6. **Activate the environment** in your compute node session:
   ```bash
   micromamba activate si_ks4_env
   ```

#### Running the Interactive Script

7. **Connect via VS Code**:
   * Press `F1` → "Remote-SSH: Connect to Host..."
   * Choose `hpc2` (or your configured host)
   * Enter your password
   * Wait for the connection to establish

8. **Open the repository**:
   * Once connected, open this repository folder from the HPC filesystem
   * Navigate to `/nfs/nhome/live/vplattner/spikesorting-pipeline`

9. **Configure the Python interpreter**:
   * Press `Ctrl+Shift+P` → "Python: Select Interpreter"
   * Choose the interpreter from your activated environment:
     `/nfs/nhome/live/vplattner/micromamba/envs/si_ks4_env/bin/python`

10. **Run the interactive script**:
    * Open `spikeinterface_script_interactive.py`
    * Run cells individually using `Shift+Enter` or the "Run Cell" button
    * The script will execute on the compute node with GPU acceleration

**Note**: Make sure your compute node session remains active while working in VS Code. If the session expires, you'll need to request a new compute node and reconnect.

#### Local Development (Alternative)

**For local development** (requires GPU and CUDA setup):

```bash
# Install environment locally
micromamba create -n si_ks4_env -f si_ks4_env.yml
micromamba activate si_ks4_env

# Run the script
python spikeinterface_script_interactive.py
```

**Note**: The interactive script is currently configured with hardcoded paths 
for a specific dataset. You'll need to modify the path variables in the script 
for your own data.

### Manual Curation with Phy

**Goal:** Interactive visualization and manual spike sorting curation using [Phy](https://github.com/cortex-lab/phy/).

Phy is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It's optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly Neuropixels probes).

#### Installation

**On HPC (recommended):**

```bash
# Create new conda environment for Phy
conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets

# Activate environment
conda activate phy2

# Install Phy development version
pip install git+https://github.com/cortex-lab/phy.git

# Optional: Install klusta/klustakwik2 for Kwik GUI
pip install klusta klustakwik2
```

**Alternative installation using environment file:**

```bash
# If the above method has issues, try the automatic install
conda env create -f environment.yml
conda activate phy2
pip install git+https://github.com/cortex-lab/phy.git
```

#### Usage

**Launch Phy Template GUI (recommended for Kilosort outputs):**

```bash
# Navigate to your sorting output directory
cd /path/to/sorting/results

# Launch the template GUI
phy template-gui params.py
```

**Launch from Python script:**

Create a `launch.py` file in your data directory:

```python
from phy.apps.template import template_gui
template_gui("params.py")
```

#### Phy Features

* **Template GUI**: Optimized for datasets sorted with Kilosort and Spyking Circus
* **Kwik GUI**: Legacy interface for datasets sorted with klusta and klustakwik2
* **Interactive visualization**: Large-scale electrophysiological data
* **Manual curation**: Refine spike sorting results
* **High-density support**: Hundreds to thousands of recording sites

#### Hardware Requirements

* **Storage**: SSD recommended for performance
* **Graphics**: Recent graphics and OpenGL drivers
* **No specific GPU requirements** for the GUI itself

#### Troubleshooting

**Common issues:**

* **PyQt5.QtWebEngineWidget error**: Run `pip install PyQtWebEngine`
* **Mac M-series chips**: Not officially supported, may require workarounds
* **Upgrading from phy 1**: Don't install phy 1 and phy 2 in the same environment

**For more help:**
* [Phy Documentation](https://phy.readthedocs.io/)
* [GitHub Issues](https://github.com/cortex-lab/phy/issues)
* [Mailing List](https://groups.google.com/forum/#!forum/phy-users)

### Batch Processing on HPC (SLURM)

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

### Results Viewing and Analysis

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

### Data Format Support

**Open Ephys:**

* TTLs in separate events file
* Geometry from `settings.xml`

**SpikeGLX:**

* TTLs as extra channel
* Geometry manual or from metadata

### Pipeline Features

* **Preprocessing:**

  * Phase shift correction
  * Bandpass filtering (300-6000 Hz)
  * Common reference (global median)

* **Spike sorting:**

  * Kilosort 4 algorithm
  * GPU-accelerated processing
  * Automatic channel grouping

* **Post-processing:**

  * Waveform extraction
  * Quality metrics computation
  * Automatic curation (empty units, excess spikes)

* **Quality metrics:**

  * Signal-to-noise ratio (SNR)
  * ISI violations ratio
  * Presence ratio
  * Firing rate
  * Number of spikes

### Kilosort GUI Not Required

This pipeline uses **Kilosort 4** through SpikeInterface's wrapper, which runs 
the sorter natively without requiring the MATLAB GUI. The `singularity_image` 
parameter controls execution:

* `singularity_image=False`: Runs natively (requires local Kilosort installation)
* `singularity_image=True`: Runs in Singularity container (requires Singularity)

### Known Issues and Limitations

* **Short recordings**: Quality metrics may be unreliable for recordings < 2-5 minutes
* **Memory usage**: Large datasets may require significant RAM
* **GPU memory**: Kilosort 4 requires sufficient GPU memory for the dataset size
* **Path dependencies**: Interactive script has hardcoded paths that need modification

### Troubleshooting

**Common issues:**

* **Import errors**: Ensure the correct Python environment is activated
* **GPU errors**: Check CUDA installation and GPU memory availability
* **Path errors**: Verify data paths in the interactive script
* **Memory errors**: Reduce dataset size or increase allocated memory

**Getting help:**

* Check the [SpikeInterface documentation](https://spikeinterface.readthedocs.io)
* Review the [Kilosort 4 documentation](https://github.com/MouseLand/Kilosort4)
* Contact the Akrami Lab for pipeline-specific issues

