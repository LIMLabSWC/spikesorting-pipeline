#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive SpikeInterface pipeline for Open Ephys recordings using
Kilosort 4. Designed to be run step-by-step (cell-by-cell) in VS Code
for testing and debugging.
"""

# %%
# ---------------------------
# Imports
# ---------------------------

# Standard Python libraries for system information and file handling
import platform  # Get system information (node name, platform, etc.)
import sys  # Access Python version and system-specific parameters
import os  # Operating system interface (working directory, environment variables)
from pathlib import (
    Path,
)  # Object-oriented filesystem paths (better than os.path)

# Scientific computing libraries
import numpy as np  # Numerical computing - arrays, math operations
import matplotlib.pyplot as plt  # Plotting library for creating figures
import xml.etree.ElementTree as ET  # Parse XML files (for Open Ephys settings)

# Error handling and debugging

# SpikeInterface core functionality
from spikeinterface import (
    extract_waveforms,
)  # Extract spike waveforms from recordings
from spikeinterface.extractors import (
    read_openephys,
)  # Load Open Ephys data files
from spikeinterface.preprocessing import (
    phase_shift,  # Correct for inter-sample timing shifts
    bandpass_filter,  # Filter signals to specific frequency range
    common_reference,  # Remove common noise across channels
    scale_to_physical_units,  # Convert to proper voltage units (µV)
)
from spikeinterface.sorters import (
    # get_default_sorter_params,
    run_sorter,
    # get_sorter_params_description,
)  # Run spike sorting algorithms
from spikeinterface.qualitymetrics import (
    compute_quality_metrics,
)  # Calculate quality metrics
from spikeinterface import (
    curation,
)  # Manual curation tools (remove bad units/spikes)
from spikeinterface.widgets import (
    plot_traces,
)  # Visualization widgets

# Probe geometry handling


def get_scaled_recording(recording):
    """
    Return scaled copy of recording (float32, in µV) for display.

    This function converts the raw recording data to proper voltage units (microvolts)
    for visualization. Raw recordings often have arbitrary units that need to be
    converted to meaningful voltage values.

    Parameters:
    -----------
    recording : BaseRecording
        The input recording object

    Returns:
    --------
    BaseRecording
        A new recording object with data scaled to µV units
    """
    return scale_to_physical_units(recording)


# %%
# ---------------------------
# System Information
# ---------------------------

# This section prints system information to help with debugging and ensure
# we're running on the correct compute node with the right environment.

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(
    f"Node: {platform.node()}"
)  # Name of the compute node (important for HPC)
print(f"Platform: {platform.platform()}")  # Operating system and architecture
print(f"Python: {sys.version.split()[0]}")  # Python version (should be 3.11)
print(f"Working directory: {os.getcwd()}")  # Current working directory
print(f"User: {os.getenv('USER', 'unknown')}")  # Username running the script
print("=" * 60)


# %%
# ---------------------------
# Paths and Flags
# ---------------------------

# This section defines all the file paths and metadata for the dataset.
# The paths follow a BIDS-like structure for organized data management.

# Define the root directory where all project data is stored
# This should point to your main data storage location
root_path = Path("/ceph/akrami/_projects/sound_cat_rat")

# Define subject/session metadata - these identify the specific recording
# These variables should be modified for each new dataset you process
subject = "sub-003_id-LP12_expmtr-lida"  # Subject identifier (animal ID)
session = "ses-05_date-20250725T114859_dtype-ephys"  # Session folder
date = "2025-07-25_11-55-01"  # Recording date and time (from Open Ephys)
experiment = "experiment1"  # Experiment name (from Open Ephys)

print("PATH CONFIGURATION")
print("-" * 40)
print(f"Root path: {root_path}")
print(f"Subject: {subject}")
print(f"Session: {session}")
print(f"Date: {date}")
print(f"Experiment: {experiment}")

# Construct the input data path - this points to the raw Open Ephys recording
# Open Ephys creates a nested folder structure: Record Node -> Experiment -> Recording
data_path = (
    root_path
    / "rawdata"
    / subject
    / session
    / "ephys"
    / date
    / "Record Node 101"
    / experiment
    / "recording1"
)

# Construct the output path - this is where all processed data will be saved
# Following BIDS derivatives structure for organized outputs
output_path = root_path / "derivatives" / subject / session / date

print(f"📂 Data path: {data_path}")
print(f"📂 Output path: {output_path}")

# Create a subdirectory for plots and ensure it exists
plot_path = output_path / "plots"
plot_path.mkdir(
    parents=True, exist_ok=True
)  # Create directory if it doesn't exist
print(f"📂 Plot path: {plot_path}")
print("-" * 40)


# %%
# ---------------------------
# Load Raw Recording and Trim
# ---------------------------

# This section loads the raw electrophysiological data and trims it to a
# manageable size for processing. Full recordings can be very large, so we
# typically work with a subset for testing and development.

print("LOADING RAW RECORDING")
print("-" * 40)
print(f"📂 Loading from: {data_path}")

# Load the Open Ephys recording - stream_id="0" refers to the first data stream
# Open Ephys can have multiple streams (e.g., continuous data, events, etc.)
raw_recording = read_openephys(data_path, stream_id="0")
print("✅ Recording loaded successfully")
print(
    f"Recording dtype: {raw_recording.get_dtype()}"
)  # Data type (usually int16 or float32)

# Define the time window to extract from the full recording
# This is useful for testing with a smaller dataset or focusing on specific time periods
start_time_sec = (
    20 * 60
)  # Start at 20 minutes into the recording (1200 seconds)
duration_sec = 30  # Extract 2 minutes (120 seconds) of data
fs = (
    raw_recording.get_sampling_frequency()
)  # Get sampling frequency (usually 30 kHz)
start_frame = int(start_time_sec * fs)  # Convert time to frame number
end_frame = int((start_time_sec + duration_sec) * fs)  # Calculate end frame

print("Trimming recording...")
print(f"   Start time: {start_time_sec}s ({start_frame} frames)")
print(f"   Duration: {duration_sec}s ({end_frame - start_frame} frames)")
print(f"   Sampling rate: {fs} Hz")

# Extract the specified time window from the full recording
# This creates a new recording object with only the selected frames
raw_recording = raw_recording.frame_slice(
    start_frame=start_frame, end_frame=end_frame
)
print(
    f"✅ Trimmed recording to {duration_sec} seconds starting from "
    f"{start_time_sec}s"
)
print("-" * 40)


# %%
# ---------------------------
# Attach Probe Geometry
# ---------------------------

# This section attaches the physical probe geometry to the recording.
# SpikeInterface needs to know where each electrode is located in 3D space
# to perform proper spike sorting and analysis. This information is stored
# in the Open Ephys settings.xml file.

print("ATTACHING PROBE GEOMETRY")
print("-" * 40)

# Import probe interface classes for creating proper probe objects

# Check if probe geometry is already attached to the recording
if "location" not in raw_recording.get_property_keys():
    print("No channel locations found — extracting from settings.xml")

    # Open Ephys stores probe geometry in a settings.xml file
    settings_path = data_path.parent.parent / "settings.xml"
    if not settings_path.exists():
        raise FileNotFoundError(f"❌ Missing settings file: {settings_path}")

    print(f"Parsing settings from: {settings_path}")
    tree = ET.parse(settings_path)
    root = tree.getroot()

    # Extract x,y coordinates for each channel from the XML
    xpos, ypos = [], []
    for ch in range(384):  # Neuropixels 1.0 has 384 channels
        x = float(root.find(".//ELECTRODE_XPOS").get(f"CH{ch}"))
        y = float(root.find(".//ELECTRODE_YPOS").get(f"CH{ch}"))
        xpos.append(x)
        ypos.append(y)

    # Combine x,y coordinates into a 2D array
    coords = np.column_stack((xpos, ypos))

    # Set the location property directly - simple and works
    raw_recording.set_property("location", coords)
    print(f"✅ Set probe locations for {len(coords)} channels")
else:
    print("✅ Probe locations already present")

print("-" * 40)


# %%
# ---------------------------
# Plot Probe Layout with Variance
# ---------------------------

# This section creates a visualization of the probe layout colored by signal variance.
# This helps identify which channels have good signal quality and which might be
# in areas with neural activity or noise.

print("PLOTTING PROBE LAYOUT WITH VARIANCE")
print("-" * 40)

# Extract a short segment of data for variance calculation
# We use 10 seconds to get a good estimate of signal variance
duration = 10
short_traces = get_scaled_recording(raw_recording).get_traces(
    start_frame=0, end_frame=int(fs * duration)
)

# Ensure data is in the correct orientation (channels x time)
# SpikeInterface can return data in different orientations depending on the extractor
if short_traces.shape[0] != raw_recording.get_num_channels():
    short_traces = short_traces.T  # Transpose if needed

# Calculate variance for each channel
# Variance is a measure of signal variability - higher variance often indicates
# neural activity or noise, while lower variance indicates quiet channels
variances = np.var(short_traces, axis=1)
locations = raw_recording.get_property("location")
print(f"✅ Variance computed for {len(variances)} channels")

# Create a color normalization for the variance values
norm_var = plt.Normalize(
    vmin=np.percentile(variances, 5), vmax=np.percentile(variances, 95)
)

# Create the plot
fig, ax = plt.subplots(figsize=(5, 12))

# Draw each electrode as a colored rectangle
for i in range(locations.shape[0]):
    x, y = locations[i]
    rect = plt.Rectangle(
        (x - 7, y - 7),
        14,
        14,
        facecolor=plt.cm.viridis(norm_var(variances[i])),
        edgecolor="gray",
        linewidth=0.5,
    )
    ax.add_patch(rect)

# Add a horizontal line at the middle of the probe for reference
y_coords = locations[:, 1]
y_min, y_max = y_coords.min(), y_coords.max()
y_mid = (y_min + y_max) / 2
ax.axhline(y=y_mid, color="black", linestyle="--", linewidth=0.8)

# Set plot properties
ax.set_aspect("equal")
ax.set_xlim(locations[:, 0].min() - 60, locations[:, 0].max() + 30)
ax.set_ylim(y_min - 20, y_max + 20)
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("Neuropixels Probe Layout (Variance Colored)")

# Add colorbar to show variance scale
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm_var)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Signal Variance (µV²)")

plt.tight_layout()
plt.savefig(plot_path / "probe_layout_colored.png", dpi=300)
plt.show()
plt.close()
print(f"💾 Saved probe layout to: {plot_path / 'probe_layout_colored.png'}")

print("-" * 40)


# %%
# ---------------------------
# Preprocessing
# ---------------------------

# This section applies standard preprocessing steps to clean the raw signal
# before spike sorting. These steps help remove noise and artifacts that
# could interfere with spike detection and sorting.

print("PREPROCESSING RECORDING")
print("-" * 40)

# Step 1: Phase shift correction
# Some recording systems have timing offsets between channels due to
# multiplexing or other hardware issues. This step corrects for those offsets.
if "inter_sample_shift" in raw_recording.get_property_keys():
    print("Applying phase shift...")
    shifted_recording = phase_shift(raw_recording)
    recording_for_filter = shifted_recording
    print("✅ Phase shift applied")
else:
    recording_for_filter = raw_recording
    print("Skipped phase shift (no inter_sample_shift property)")

# Step 2: Bandpass filtering
# Remove low-frequency drift and high-frequency noise by keeping only
# frequencies relevant to neural spikes (300-6000 Hz)
print("Applying bandpass filter (300-6000 Hz)...")
filtered_recording = bandpass_filter(
    recording_for_filter, freq_min=300, freq_max=6000
)
print("✅ Bandpass filter applied")

# Step 3: Common reference (noise removal)
# Remove common noise across all channels by subtracting the median
# of all channels from each channel. This helps remove environmental
# noise that affects all electrodes similarly.
channel_group = filtered_recording.get_property("group")
channel_ids = filtered_recording.get_channel_ids()

# Determine how to group channels for common reference
# For single-shank probes, we typically use all channels together
if channel_group is None:
    split_channel_ids = [channel_ids.tolist()]
    print(f"No 'group' property — using all {len(channel_ids)} channels")
else:
    # For multi-shank probes, we might want to reference each shank separately
    split_channel_ids = [
        channel_ids[channel_group == idx].tolist()
        for idx in np.unique(channel_group)
    ]
    print(f"Found {len(split_channel_ids)} channel groups")

print("Applying common reference (global median)...")
preprocessed_recording = common_reference(
    filtered_recording,
    reference="global",  # Use all channels as reference
    operator="median",  # Use median instead of mean (more robust to outliers)
    groups=split_channel_ids,  # Channel groups to process separately
)
print("✅ Common reference applied")
print("-" * 40)


# %%
# ---------------------------
# Plot Preprocessed Signal
# ---------------------------

print("PLOTTING PREPROCESSED SIGNAL")
print("-" * 40)
print("Creating timeseries map...")

plt.figure(figsize=(24, 28))
plot_traces(
    get_scaled_recording(preprocessed_recording),
    time_range=(1201, 1202),
    order_channel_by_depth=True,
    show_channel_ids=False,
    mode="map",
    return_scaled=True,
    clim=(-150, 150),
)
plt.xlabel("Time (s)")
plt.ylabel("Depth (µm)")
plt.title("Preprocessed Signal Map")
plt.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.4)
plt.tight_layout()
plt.savefig(plot_path / "preprocessing_full.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(
    f"💾 Saved preprocessed signal map to: {plot_path / 'preprocessing_full.png'}"
)
print("-" * 40)


# %%
# ---------------------------
# Plot RMS Histogram
# ---------------------------

print("COMPUTING RMS PER CHANNEL")
print("-" * 40)
print(f"Computing RMS for {duration}s of data...")

duration = 10
n_channels = preprocessed_recording.get_num_channels()

traces = get_scaled_recording(preprocessed_recording).get_traces(
    start_frame=0, end_frame=int(fs * duration)
)

if traces.shape[0] != n_channels:
    traces = traces.T

rms_values = np.sqrt(np.mean(traces**2, axis=1))
print(f"✅ RMS computed for {len(rms_values)} channels")

print("Creating RMS histogram...")
plt.figure(figsize=(6, 4))
plt.hist(rms_values, bins=50, color="skyblue", edgecolor="black")
plt.title("RMS Distribution Across Channels")
plt.xlabel("RMS (µV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plot_path / "rms_histogram.png", dpi=200)
plt.show()
plt.close()
print(f"💾 Saved RMS histogram to: {plot_path / 'rms_histogram.png'}")
print("-" * 40)


# %%
# ---------------------------
# Run Kilosort 4
# ---------------------------

# This section runs the Kilosort 4 spike sorting algorithm on the preprocessed data.
# Kilosort 4 is a state-of-the-art spike sorting algorithm that uses template
# matching and clustering to identify individual neurons from multi-electrode data.

print("RUNNING KILOSORT 4")
print("-" * 40)

# -------------------------------------
# Assign a default 'group' property if missing
# -------------------------------------
# The Kilosort4 wrapper in SpikeInterface expects a 'group' property
# for channel grouping. This is mainly used to handle multi-shank probes
# or multiple probes by sorting them separately.
#
# In our case (a single-shank Neuropixels probe), all channels come from
# one shank, so no grouping is needed. However, Kilosort4 still expects
# the 'group' field to exist.
#
# We assign all channels to group 0 using a flat array of zeros.
print("Setting up channel groups...")
channel_ids = preprocessed_recording.get_channel_ids()
preprocessed_recording.set_property(
    "group", np.zeros(len(channel_ids), dtype=int)
)
print(f"✅ All {len(channel_ids)} channels assigned to group 0")

# -------------------------------------
# Run spike sorting with Kilosort4
# -------------------------------------
# This calls the SpikeInterface wrapper for Kilosort4 and launches the
# sorter natively (not in a container).
#
# The 'singularity_image' flag controls how the sorter is executed:
# - If True: sorter runs inside a Singularity container.
#            You don’t need to install MATLAB or Kilosort locally,
#            but Singularity must be configured correctly.
# - If False: sorter runs natively — so Kilosort4 must already be
#             installed and compiled on your system, and the MATLAB
#             runtime or CLI must be available.
#
# The sorting output will be saved in the given output path.
print("Starting Kilosort 4 spike sorting...")
print(f"📂 Output directory: {output_path / 'sorting'}")
print("This may take several minutes...")

sorting = run_sorter(
    "kilosort4",  # Sorter name
    preprocessed_recording,  # Input recording (already filtered)
    (output_path / "sorting").as_posix(),  # Output directory for sorting files
    singularity_image=False,  # Run sorter natively
    remove_existing_folder=True,  # Overwrite if folder exists
)

print("✅ Kilosort 4 completed successfully!")
print("-" * 40)


# %%
# ---------------------------
# Curate and Extract Waveforms
# ---------------------------

# This section performs post-sorting curation and extracts spike waveforms.
# Curation removes obviously bad units, and waveform extraction provides
# the data needed for quality metrics and visualization.

print("CURATING AND EXTRACTING WAVEforms")
print("-" * 40)

# Step 1: Remove empty units
# Some units might have no spikes after sorting - remove these
print("Removing empty units...")
sorting = sorting.remove_empty_units()

# Step 2: Remove excess spikes
# Sometimes spike sorters detect the same spike multiple times
# This step removes duplicate spikes that are too close in time
print("Removing excess spikes...")
sorting = curation.remove_excess_spikes(sorting, preprocessed_recording)

# Step 3: Extract waveforms
# Extract the actual spike waveforms around each detected spike
# This is needed for quality metrics and visualization
print("Extracting waveforms...")
print(
    "Note: You may see a warning about 'recording will not be persistent on disk' - this is normal for in-memory recordings"
)

waveforms = extract_waveforms(
    preprocessed_recording,  # Input recording
    sorting,  # Spike sorting results
    folder=(output_path / "postprocessing").as_posix(),  # Output folder
    ms_before=2,  # Extract 2ms before spike peak
    ms_after=2,  # Extract 2ms after spike peak
    max_spikes_per_unit=500,  # Limit spikes per unit (for memory)
    return_scaled=True,  # Return in µV units
    sparse=True,  # Only extract from nearby channels
    peak_sign="neg",  # Look for negative peaks (typical for extracellular)
    method="radius",  # Use radius-based channel selection
    radius_um=75,  # Extract from channels within 75µm
)
print("✅ Waveforms extracted successfully")
print("-" * 40)


# %%
# ---------------------------
# Compute Quality Metrics
# ---------------------------

# This section computes various quality metrics for each sorted unit.
# These metrics help assess the quality of spike sorting and identify
# which units are likely to represent single neurons.

print("COMPUTING QUALITY METRICS")
print("-" * 40)
print("Computing quality metrics...")

# Compute comprehensive quality metrics for all units
# This includes metrics like:
# - SNR (Signal-to-Noise Ratio): How well the spike stands out from noise
# - ISI violations: Whether spikes occur too close together (violating refractory period)
# - Presence ratio: How consistently the unit fires throughout the recording
# - Firing rate: Average spikes per second
# - And many more...
quality_metrics = compute_quality_metrics(waveforms)

# Save the quality metrics to a CSV file for later analysis
# This file can be loaded in Python, Excel, or other analysis tools
quality_metrics.to_csv(output_path / "postprocessing" / "quality_metrics.csv")

print("✅ Quality metrics computed successfully")
print(
    f"💾 Metrics saved to: {output_path / 'postprocessing' / 'quality_metrics.csv'}"
)
print(f"💾 Plots saved to: {plot_path}")
print("=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
