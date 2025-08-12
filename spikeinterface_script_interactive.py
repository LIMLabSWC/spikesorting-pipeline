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

import platform
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import traceback

# Lightweight package for nice tracebacks
try:
    import rich.traceback
    rich.traceback.install(show_locals=True)
    print("✓ Rich traceback installed")
except ImportError:
    print("⚠ Rich not available, using standard traceback")

from spikeinterface import extract_waveforms
from spikeinterface.extractors import read_openephys
from spikeinterface.preprocessing import (
    phase_shift, bandpass_filter, common_reference,
    scale_to_physical_units
)
from spikeinterface.sorters import run_sorter
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface import curation
from spikeinterface.widgets import plot_timeseries, plot_traces

from probeinterface.plotting import plot_probe


def get_scaled_recording(recording):
    """Return scaled copy of recording (float32, in µV) for display."""
    return scale_to_physical_units(recording)


# %%
# ---------------------------
# System Information
# ---------------------------

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"Node: {platform.node()}")
print(f"Platform: {platform.platform()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Working directory: {os.getcwd()}")
print(f"User: {os.getenv('USER', 'unknown')}")
print("=" * 60)


# %%
# ---------------------------
# Paths and Flags
# ---------------------------

# Define root
root_path = Path("/ceph/akrami/_projects/sound_cat_rat")

# Define subject/session info
subject = "sub-003_id-LP12_expmtr-lida"
session = "ses-04_date-20250724T132332_dtype-ephys"
date = "2025-07-24_13-24-43"
experiment = "experiment1"

print("PATH CONFIGURATION")
print("-" * 40)
print(f"Root path: {root_path}")
print(f"Subject: {subject}")
print(f"Session: {session}")
print(f"Date: {date}")
print(f"Experiment: {experiment}")

# Construct data path using parentheses for clean line continuation
data_path = (
    root_path / "rawdata" / subject / session / "ephys" / date
    / "Record Node 101" / experiment / "recording1"
)

# Construct output path (cleaner f-string usage is optional here)
output_path = root_path / "derivatives" / subject / session / date

print(f"📂 Data path: {data_path}")
print(f"📂 Output path: {output_path}")

plot_path = output_path / "plots"
plot_path.mkdir(parents=True, exist_ok=True)
print(f"📂 Plot path: {plot_path}")
print("-" * 40)


# %%
# ---------------------------
# Load Raw Recording and Trim
# ---------------------------

print("LOADING RAW RECORDING")
print("-" * 40)
print(f"📂 Loading from: {data_path}")
raw_recording = read_openephys(data_path, stream_id="0")
print(f"✅ Recording loaded successfully")
print(f"Recording dtype: {raw_recording.get_dtype()}")

start_time_sec = 20 * 60
duration_sec = 120
fs = raw_recording.get_sampling_frequency()
start_frame = int(start_time_sec * fs)
end_frame = int((start_time_sec + duration_sec) * fs)

print("Trimming recording...")
print(f"   Start time: {start_time_sec}s ({start_frame} frames)")
print(f"   Duration: {duration_sec}s ({end_frame - start_frame} frames)")
print(f"   Sampling rate: {fs} Hz")

raw_recording = raw_recording.frame_slice(
    start_frame=start_frame, end_frame=end_frame
)
print(f"✅ Trimmed recording to {duration_sec} seconds starting from "
      f"{start_time_sec}s")
print("-" * 40)


# %%
# ---------------------------
# Attach Probe Geometry
# ---------------------------

print("ATTACHING PROBE GEOMETRY")
print("-" * 40)

if "location" not in raw_recording.get_property_keys():
    print("No channel locations found — extracting from settings.xml")
    settings_path = data_path.parent.parent / "settings.xml"
    if not settings_path.exists():
        raise FileNotFoundError(f"❌ Missing settings file: {settings_path}")

    print(f"Parsing settings from: {settings_path}")
    tree = ET.parse(settings_path)
    root = tree.getroot()
    xpos, ypos = [], []
    for ch in range(384):
        x = float(root.find(".//ELECTRODE_XPOS").get(f"CH{ch}"))
        y = float(root.find(".//ELECTRODE_YPOS").get(f"CH{ch}"))
        xpos.append(x)
        ypos.append(y)

    coords = np.column_stack((xpos, ypos))
    raw_recording.set_property("location", coords)
    print(f"✅ Extracted coordinates for {len(coords)} channels")
else:
    print("✅ Probe locations already present")
print("-" * 40)


# %%
# ---------------------------
# Plot Probe Layout with Variance
# ---------------------------

print("PLOTTING PROBE LAYOUT WITH VARIANCE")
print("-" * 40)
duration = 10
short_traces = get_scaled_recording(raw_recording).get_traces(
    start_frame=0, end_frame=int(fs * duration)
)
if short_traces.shape[0] != raw_recording.get_num_channels():
    short_traces = short_traces.T

variances = np.var(short_traces, axis=1)
locations = raw_recording.get_property("location")
print(f"✅ Variance computed for {len(variances)} channels")

norm_var = plt.Normalize(
    vmin=np.percentile(variances, 5),
    vmax=np.percentile(variances, 95)
)

fig, ax = plt.subplots(figsize=(5, 12))
for i in range(locations.shape[0]):
    x, y = locations[i]
    rect = plt.Rectangle(
        (x - 7, y - 7), 14, 14,
        facecolor=plt.cm.viridis(norm_var(variances[i])),
        edgecolor='gray', linewidth=0.5
    )
    ax.add_patch(rect)

y_coords = locations[:, 1]
y_min, y_max = y_coords.min(), y_coords.max()
y_mid = (y_min + y_max) / 2
ax.axhline(y=y_mid, color='black', linestyle='--', linewidth=0.8)

ax.set_aspect("equal")
ax.set_xlim(locations[:, 0].min() - 60, locations[:, 0].max() + 30)
ax.set_ylim(y_min - 20, y_max + 20)
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("Neuropixels Probe Layout (Variance Colored)")

sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm_var)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Signal Variance (µV²)")

plt.tight_layout()
plt.savefig(plot_path / "probe_layout_colored.png", dpi=300)
plt.close()
print(f"💾 Saved probe layout to: {plot_path / 'probe_layout_colored.png'}")
print("-" * 40)


# %%
# ---------------------------
# Preprocessing
# ---------------------------

print("PREPROCESSING RECORDING")
print("-" * 40)

# Phase shift
if "inter_sample_shift" in raw_recording.get_property_keys():
    print("Applying phase shift...")
    shifted_recording = phase_shift(raw_recording)
    recording_for_filter = shifted_recording
    print("✅ Phase shift applied")
else:
    recording_for_filter = raw_recording
    print("Skipped phase shift (no inter_sample_shift property)")

# Bandpass filtering
print("Applying bandpass filter (300-6000 Hz)...")
filtered_recording = bandpass_filter(
    recording_for_filter, freq_min=300, freq_max=6000
)
print("✅ Bandpass filter applied")

# Common reference
channel_group = filtered_recording.get_property("group")
channel_ids = filtered_recording.get_channel_ids()

if channel_group is None:
    split_channel_ids = [channel_ids.tolist()]
    print(f"No 'group' property — using all {len(channel_ids)} channels")
else:
    split_channel_ids = [
        channel_ids[channel_group == idx].tolist()
        for idx in np.unique(channel_group)
    ]
    print(f"Found {len(split_channel_ids)} channel groups")

print("Applying common reference (global median)...")
preprocessed_recording = common_reference(
    filtered_recording,
    reference="global",
    operator="median",
    groups=split_channel_ids,
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
plt.savefig(plot_path / "preprocessing_full.png", dpi=200,
            bbox_inches="tight")
plt.close()
print(f"💾 Saved preprocessed signal map to: {plot_path / 'preprocessing_full.png'}")
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

rms_values = np.sqrt(np.mean(traces ** 2, axis=1))
print(f"✅ RMS computed for {len(rms_values)} channels")

print("Creating RMS histogram...")
plt.figure(figsize=(6, 4))
plt.hist(rms_values, bins=50, color="skyblue", edgecolor="black")
plt.title("RMS Distribution Across Channels")
plt.xlabel("RMS (µV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plot_path / "rms_histogram.png", dpi=200)
plt.close()
print(f"💾 Saved RMS histogram to: {plot_path / 'rms_histogram.png'}")
print("-" * 40)


# %% 
# ---------------------------
# Run Kilosort 4
# ---------------------------

print("RUNNING KILOSORT 4")
print("-" * 40)

# -------------------------------------
# Assign a default 'group' property if missing
# -------------------------------------
# The Kilosort4 wrapper in SpikeInterface expects a 'group' property 
# for channel grouping. This is mainly used to handle multi-shank probes 
# or multiple probes by sorting them separately.
# 
# In your case (a single-shank Neuropixels probe), all channels come from 
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
    "kilosort4",                             # Sorter name
    preprocessed_recording,                  # Input recording (already filtered)
    (output_path / "sorting").as_posix(),    # Output directory for sorting files
    singularity_image=False,                 # Run sorter natively
    remove_existing_folder=True              # Overwrite if folder exists
)

print("✅ Kilosort 4 completed successfully!")
print("-" * 40)



# %%
# ---------------------------
# Curate and Extract Waveforms
# ---------------------------

print("CURATING AND EXTRACTING WAVEforms")
print("-" * 40)

print("Removing empty units...")
sorting = sorting.remove_empty_units()
print("Removing excess spikes...")
sorting = curation.remove_excess_spikes(
    sorting, preprocessed_recording
)

print("Extracting waveforms...")
waveforms = extract_waveforms(
    preprocessed_recording,
    sorting,
    folder=(output_path / "postprocessing").as_posix(),
    ms_before=2,
    ms_after=2,
    max_spikes_per_unit=500,
    return_scaled=True,
    sparse=True,
    peak_sign="neg",
    method="radius",
    radius_um=75,
)
print("✅ Waveforms extracted successfully")
print("-" * 40)


# %%
# ---------------------------
# Compute Quality Metrics
# ---------------------------

print("COMPUTING QUALITY METRICS")
print("-" * 40)
print("Computing quality metrics...")
quality_metrics = compute_quality_metrics(waveforms)
quality_metrics.to_csv(
    output_path / "postprocessing" / "quality_metrics.csv"
)

print("✅ Quality metrics computed successfully")
print(f"💾 Metrics saved to: {output_path / 'postprocessing' / 'quality_metrics.csv'}")
print(f"💾 Plots saved to: {plot_path}")
print("=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
