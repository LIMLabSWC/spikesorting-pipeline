#!/usr/bin/env python3
"""
SpikeInterface pipeline for Open Ephys recordings using Kilosort 4
Includes preprocessing, spike sorting, waveform extraction, and quality metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from spikeinterface import extract_waveforms
from spikeinterface.extractors import read_openephys
from spikeinterface.preprocessing import phase_shift, bandpass_filter, common_reference
from spikeinterface.sorters import run_sorter
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface import curation
from spikeinterface.widgets import plot_timeseries
from probeinterface.plotting import plot_probe

# ------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="SpikeInterface pipeline with Kilosort 4")
parser.add_argument("data_path", help="Path to the Open Ephys recording folder")
parser.add_argument(
    "--output_path",
    default="./spike_sorting_output",
    help="Output directory for results (default: ./spike_sorting_output)"
)
parser.add_argument(
    "--show_probe",
    action="store_true",
    help="Save probe layout plot (default: False)"
)
parser.add_argument(
    "--show_preprocessing",
    action="store_true",
    help="Save preprocessing traces plot (default: False)"
)
args = parser.parse_args()

# ------------------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------------------

data_path = Path(args.data_path).resolve()
output_path = Path(args.output_path).resolve()
plot_path = output_path / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

show_probe = args.show_probe
show_preprocessing = args.show_preprocessing

# ------------------------------------------------------------
# LOAD RECORDING
# ------------------------------------------------------------

print(f"Loading recording from: {data_path}")
raw_recording = read_openephys(data_path, stream_id='0')

# ------------------------------------------------------------
# ATTACH PROBE LOCATIONS (PARSE XML IF NEEDED)
# ------------------------------------------------------------

if "location" not in raw_recording.get_property_keys():
    print("No channel locations found — extracting from settings.xml")

    # Locate settings.xml two levels up from binary files
    settings_path = data_path.parent.parent / "settings.xml"
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.xml not found at {settings_path}")

    # Parse XML
    tree = ET.parse(settings_path)
    root = tree.getroot()

    xpos, ypos = [], []
    for ch in range(384):  # first 384 channels only
        x = float(root.find(f".//ELECTRODE_XPOS").get(f"CH{ch}"))
        y = float(root.find(f".//ELECTRODE_YPOS").get(f"CH{ch}"))
        xpos.append(x)
        ypos.append(y)

    coords = np.column_stack((xpos, ypos))
    raw_recording.set_property("location", coords)
else:
    print("Probe locations already present in recording.")

# Optional: Save probe layout
if show_probe:
    plot_probe(raw_recording.get_probe())
    plt.savefig(plot_path / "probe_layout.png", dpi=200, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# PHASE SHIFT CORRECTION
# ------------------------------------------------------------

if "inter_sample_shift" in raw_recording.get_property_keys():
    shifted_recording = phase_shift(raw_recording)
    recording_for_filter = shifted_recording
    print("Applied phase shift (inter_sample_shift property found).")
else:
    recording_for_filter = raw_recording
    print("Skipped phase shift (no inter_sample_shift property).")

# ------------------------------------------------------------
# BANDPASS FILTERING
# ------------------------------------------------------------

filtered_recording = bandpass_filter(recording_for_filter, freq_min=300, freq_max=6000)

# ------------------------------------------------------------
# COMMON AVERAGE REFERENCING (CAR)
# ------------------------------------------------------------

channel_group = filtered_recording.get_property("group")
channel_ids = filtered_recording.get_channel_ids()

if channel_group is None:
    split_channel_ids = [channel_ids.tolist()]
    print(f"No 'group' property found — using all {len(channel_ids)} channels as one group.")
else:
    split_channel_ids = [
        channel_ids[channel_group == idx].tolist()
        for idx in np.unique(channel_group)
    ]
    print(f"Detected {len(split_channel_ids)} groups for referencing.")

preprocessed_recording = common_reference(
    filtered_recording,
    reference="global",
    operator="median",
    groups=split_channel_ids
)

# ------------------------------------------------------------
# SAVE PREPROCESSING PLOTS
# ------------------------------------------------------------

if show_preprocessing:
    if "group" in preprocessed_recording.get_property_keys():
        recs_grouped_by_shank = preprocessed_recording.split_by("group")
        for g_idx, rec in enumerate(recs_grouped_by_shank):
            plot_timeseries(
                preprocessed_recording,
                order_channel_by_depth=True,
                time_range=(3499, 3500),
                return_scaled=True,
                show_channel_ids=True,
                mode="map",
            )
            plt.savefig(plot_path / f"preprocessing_group{g_idx}.png", dpi=200, bbox_inches="tight")
            plt.close()
    else:
        plot_timeseries(
            preprocessed_recording,
            order_channel_by_depth=True,
            time_range=(3499, 3500),
            return_scaled=True,
            show_channel_ids=True,
            mode="map",
        )
        plt.savefig(plot_path / "preprocessing_full.png", dpi=200, bbox_inches="tight")
        plt.close()

# ------------------------------------------------------------
# RUN KILOSORT 4 SPIKE SORTING
# ------------------------------------------------------------

print("Running Kilosort 4...")
sorting = run_sorter(
    "kilosort4",
    preprocessed_recording,
    (output_path / "sorting").as_posix(),
    singularity_image=True,     # adjust if using singularity on HPC
    car=False,
    freq_min=150,
    remove_existing_folder=True
)

# ------------------------------------------------------------
# CURATE SORTING AND EXTRACT WAVEFORMS
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# COMPUTE QUALITY METRICS
# ------------------------------------------------------------

print("Computing quality metrics...")
quality_metrics = compute_quality_metrics(waveforms)
quality_metrics.to_csv(output_path / "postprocessing")

print("Quality metrics saved to:", output_path / "postprocessing")
print("Plots saved to:", plot_path)
