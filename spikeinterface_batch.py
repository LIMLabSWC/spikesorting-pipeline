#!/usr/bin/env python3
"""
SpikeInterface batch pipeline for Open Ephys recordings using Kilosort 4.
This version is SLURM-compatible and accepts CLI arguments for flexible deployment.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from spikeinterface import extract_waveforms
from spikeinterface.extractors import read_openephys
from spikeinterface.preprocessing import (
    phase_shift,
    bandpass_filter,
    common_reference,
    scale_to_physical_units,
)
from spikeinterface.sorters import run_sorter
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface import curation
from spikeinterface.widgets import plot_traces
from probeinterface.plotting import plot_probe


def get_scaled_recording(recording):
    return scale_to_physical_units(recording)


# ---------------------------
# Argument Parsing
# ---------------------------

parser = argparse.ArgumentParser(
    description="SpikeInterface SLURM-compatible pipeline with Kilosort 4"
)
parser.add_argument("data_path", help="Path to the Open Ephys recording folder")
parser.add_argument(
    "--output_path",
    default="./spike_sorting_output",
    help="Output directory for results",
)
parser.add_argument(
    "--show_probe", action="store_true", help="Plot probe layout"
)
parser.add_argument(
    "--show_preprocessing",
    action="store_true",
    help="Plot preprocessing traces",
)
args = parser.parse_args()

# ---------------------------
# Paths
# ---------------------------

data_path = Path(args.data_path).resolve()
output_path = Path(args.output_path).resolve()
plot_path = output_path / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load and Trim Recording (full duration)
# ---------------------------

print(f"Loading recording from: {data_path}")
raw_recording = read_openephys(data_path, stream_id="0")
fs = raw_recording.get_sampling_frequency()
print(f"Recording dtype: {raw_recording.get_dtype()}")
print(f"Duration: {raw_recording.get_num_frames() / fs:.1f} seconds")

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

if args.show_probe:
    print("Saving probe layout plot...")
    plot_probe(raw_recording.get_probe())
    plt.savefig(plot_path / "probe_layout.png", dpi=300, bbox_inches="tight")
    plt.close()

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

# ---------------------------
# Plot Preprocessed Signal (Optional)
# ---------------------------

if args.show_preprocessing:
    print("Plotting preprocessed signal...")
    plt.figure(figsize=(20, 24))
    plot_traces(
        get_scaled_recording(preprocessed_recording),
        time_range=(10, 11),
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
    plt.savefig(
        plot_path / "preprocessing_full.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

# ---------------------------
# Run Kilosort 4
# ---------------------------

print("Running Kilosort 4...")
preprocessed_recording.set_property(
    "group", np.zeros(len(channel_ids), dtype=int)
)
sorting = run_sorter(
    "kilosort4",
    preprocessed_recording,
    (output_path / "sorting").as_posix(),
    singularity_image=True,
    remove_existing_folder=True,
)

# ---------------------------
# Curate and Extract Waveforms
# ---------------------------

sorting = sorting.remove_empty_units()
sorting = curation.remove_excess_spikes(sorting, preprocessed_recording)

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

# ---------------------------
# Compute Quality Metrics
# ---------------------------

print("Computing quality metrics...")
quality_metrics = compute_quality_metrics(waveforms)
quality_metrics.to_csv(output_path / "postprocessing" / "quality_metrics.csv")

print(
    "Done. Quality metrics saved to:",
    output_path / "postprocessing" / "quality_metrics.csv",
)
print("All plots saved to:", plot_path)
