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

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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
# Paths and Flags
# ---------------------------

# Define root
root_path = Path("/ceph/akrami/_projects/sound_cat_rat")

# Define subject/session info
subject = "sub-003_id-LP12_expmtr-lida"
session = "ses-17_date-20250808T124058_dtype-ephys"
date = "2025-08-08_12-42-08"
experiment = "experiment4"

# Construct data path using parentheses for clean line continuation
data_path = (
    root_path / "rawdata" / subject / session / "ephys" / date
    / "Record Node 101" / experiment / "recording1"
)

# Construct output path (cleaner f-string usage is optional here)
output_path = root_path / "derivatives" / subject / session / date


plot_path = output_path / "plots"
plot_path.mkdir(parents=True, exist_ok=True)


# %%
# ---------------------------
# Load Raw Recording and Trim
# ---------------------------

print(f"Loading recording from: {data_path}")
raw_recording = read_openephys(data_path, stream_id="0")
print(f"Recording dtype: {raw_recording.get_dtype()}")

start_time_sec = 20 * 60
duration_sec = 30
fs = raw_recording.get_sampling_frequency()
start_frame = int(start_time_sec * fs)
end_frame = int((start_time_sec + duration_sec) * fs)

raw_recording = raw_recording.frame_slice(
    start_frame=start_frame, end_frame=end_frame
)
print(f"Trimmed recording to {duration_sec} seconds from {start_time_sec}s.")


# %%
# ---------------------------
# Attach Probe Geometry
# ---------------------------

if "location" not in raw_recording.get_property_keys():
    print("No channel locations — extracting from settings.xml")
    settings_path = data_path.parent.parent / "settings.xml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Missing: {settings_path}")

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
else:
    print("Probe locations already present.")


# %%
# ---------------------------
# Plot Probe Layout with Variance
# ---------------------------

print("Plotting Neuropixels-style probe layout...")
duration = 10
short_traces = get_scaled_recording(raw_recording).get_traces(
    start_frame=0, end_frame=int(fs * duration)
)
if short_traces.shape[0] != raw_recording.get_num_channels():
    short_traces = short_traces.T

variances = np.var(short_traces, axis=1)
locations = raw_recording.get_property("location")

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


# %%
# ---------------------------
# Preprocessing
# ---------------------------

if "inter_sample_shift" in raw_recording.get_property_keys():
    shifted_recording = phase_shift(raw_recording)
    recording_for_filter = shifted_recording
    print("Applied phase shift.")
else:
    recording_for_filter = raw_recording
    print("Skipped phase shift.")

filtered_recording = bandpass_filter(
    recording_for_filter, freq_min=300, freq_max=6000
)

channel_group = filtered_recording.get_property("group")
channel_ids = filtered_recording.get_channel_ids()

if channel_group is None:
    split_channel_ids = [channel_ids.tolist()]
    print(f"No 'group' property — using all {len(channel_ids)} channels.")
else:
    split_channel_ids = [
        channel_ids[channel_group == idx].tolist()
        for idx in np.unique(channel_group)
    ]
    print(f"Found {len(split_channel_ids)} channel groups.")

preprocessed_recording = common_reference(
    filtered_recording,
    reference="global",
    operator="median",
    groups=split_channel_ids,
)


# %%
# ---------------------------
# Plot Preprocessed Signal
# ---------------------------

print("Plotting preprocessed signal (timeseries map)...")

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


# %%
# ---------------------------
# Plot RMS Histogram
# ---------------------------

print("Computing RMS per channel...")

duration = 10
n_channels = preprocessed_recording.get_num_channels()

traces = get_scaled_recording(preprocessed_recording).get_traces(
    start_frame=0, end_frame=int(fs * duration)
)

if traces.shape[0] != n_channels:
    traces = traces.T

rms_values = np.sqrt(np.mean(traces ** 2, axis=1))

plt.figure(figsize=(6, 4))
plt.hist(rms_values, bins=50, color="skyblue", edgecolor="black")
plt.title("RMS Distribution Across Channels")
plt.xlabel("RMS (µV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plot_path / "rms_histogram.png", dpi=200)
plt.close()


# %% 
# ---------------------------
# Run Kilosort 4
# ---------------------------

print("Running Kilosort 4...")

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
channel_ids = preprocessed_recording.get_channel_ids()
preprocessed_recording.set_property(
    "group", np.zeros(len(channel_ids), dtype=int)
)

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
sorting = run_sorter(
    "kilosort4",                             # Sorter name
    preprocessed_recording,                  # Input recording (already filtered)
    (output_path / "sorting").as_posix(),    # Output directory for sorting files
    singularity_image=False,                 # Run sorter natively
    remove_existing_folder=True              # Overwrite if folder exists
)



# %%
# ---------------------------
# Curate and Extract Waveforms
# ---------------------------

sorting = sorting.remove_empty_units()
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


# %%
# ---------------------------
# Compute Quality Metrics
# ---------------------------

print("Computing quality metrics...")
quality_metrics = compute_quality_metrics(waveforms)
quality_metrics.to_csv(
    output_path / "postprocessing" / "quality_metrics.csv"
)

print("Metrics saved to:",
      output_path / "postprocessing" / "quality_metrics.csv")
print("Plots saved to:", plot_path)
