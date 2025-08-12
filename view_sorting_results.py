#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script for viewing SpikeInterface sorting results.
Can be run on any sorting result folder created by the main pipeline.

Usage:
    python view_sorting_results.py /path/to/sorting/results/folder
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from spikeinterface import load_waveforms
from spikeinterface.core import load
from spikeinterface.widgets import (
    plot_unit_waveforms, plot_unit_templates, plot_rasters,
    plot_autocorrelograms, plot_crosscorrelograms, plot_unit_locations
)


def load_sorting_results(results_path):
    """
    Load sorting results from a results folder.
    
    Parameters:
    -----------
    results_path : str or Path
        Path to the sorting results folder
        
    Returns:
    --------
    sorting : BaseSorting
        Loaded sorting object
    waveforms : WaveformExtractor
        Loaded waveforms object
    quality_df : pandas.DataFrame
        Quality metrics dataframe
    """
    results_path = Path(results_path)
    
    # Check if required files exist
    sorting_path = results_path / "sorting"
    waveforms_path = results_path / "postprocessing"
    quality_path = results_path / "postprocessing" / "quality_metrics.csv"
    
    if not sorting_path.exists():
        raise FileNotFoundError(f"Sorting folder not found: {sorting_path}")
    if not waveforms_path.exists():
        raise FileNotFoundError(f"Waveforms folder not found: {waveforms_path}")
    if not quality_path.exists():
        raise FileNotFoundError(f"Quality metrics file not found: {quality_path}")
    
    print(f"Loading sorting results from: {results_path}")
    
    # Load sorting and waveforms
    sorting = load(sorting_path)
    waveforms = load_waveforms(waveforms_path, sorting=sorting)
    quality_df = pd.read_csv(quality_path)
    
    print(f"Loaded {len(sorting.get_unit_ids())} units")
    print(f"Recording duration: {sorting.get_total_duration():.1f} seconds")
    
    return sorting, waveforms, quality_df


def create_plots_directory(results_path):
    """Create plots directory if it doesn't exist."""
    plot_path = results_path / "plots"
    plot_path.mkdir(exist_ok=True)
    return plot_path


def print_summary_statistics(sorting, quality_df):
    """Print summary statistics of the sorting results."""
    print(f"\n=== SPIKE SORTING RESULTS SUMMARY ===")
    print(f"Total units found: {len(sorting.get_unit_ids())}")
    
    # Calculate total spikes across all units
    total_spikes = sum(len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.get_unit_ids())
    print(f"Total spikes: {total_spikes}")
    print(f"Recording duration: {sorting.get_total_duration():.1f} seconds")
    print(f"Mean firing rate: {total_spikes / sorting.get_total_duration():.2f} Hz")
    
    # Warn about short recordings
    if sorting.get_total_duration() < 60:
        print(f"\n⚠️  WARNING: Short recording ({sorting.get_total_duration():.1f}s)")
        print("   - Quality metrics may be unreliable")
        print("   - Presence ratio calculations may fail")
        print("   - Consider using longer recordings (2-5 minutes minimum)")
    elif sorting.get_total_duration() < 300:
        print(f"\n⚠️  NOTE: Relatively short recording ({sorting.get_total_duration():.1f}s)")
        print("   - Consider using longer recordings for more reliable metrics")

    print(f"\n=== QUALITY METRICS SUMMARY ===")
    print(f"Mean SNR: {quality_df['snr'].mean():.2f} ± {quality_df['snr'].std():.2f}")
    print(f"Mean ISI violation rate: {quality_df['isi_violations_ratio'].mean():.3f} ± {quality_df['isi_violations_ratio'].std():.3f}")
    
    # Handle potential NaN values in presence ratio
    if 'presence_ratio' in quality_df.columns:
        presence_mean = quality_df['presence_ratio'].mean()
        presence_std = quality_df['presence_ratio'].std()
        if pd.isna(presence_mean):
            print(f"Mean presence ratio: NaN (check quality metrics)")
        else:
            print(f"Mean presence ratio: {presence_mean:.3f} ± {presence_std:.3f}")
    else:
        print("Presence ratio not available in quality metrics")

    # Show units with good quality (handle NaN values)
    if 'presence_ratio' in quality_df.columns:
        good_units = quality_df[
            (quality_df['snr'] > 5) & 
            (quality_df['isi_violations_ratio'] < 0.1) & 
            (quality_df['presence_ratio'] > 0.8)
        ]
    else:
        good_units = quality_df[
            (quality_df['snr'] > 5) & 
            (quality_df['isi_violations_ratio'] < 0.1)
        ]
    print(f"High-quality units (SNR>5, ISI<0.1, presence>0.8): {len(good_units)}")


def plot_unit_waveforms_figure(waveforms, sorting, plot_path):
    """Plot unit waveforms and templates."""
    print("Plotting unit waveforms...")
    plt.figure(figsize=(15, 10))
    plot_unit_waveforms(
        waveforms,
        unit_ids=sorting.get_unit_ids()[:min(12, len(sorting.get_unit_ids()))],
        plot_templates=True,
        same_axis=True
    )
    plt.suptitle("Unit Waveforms and Templates", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path / "unit_waveforms.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path / 'unit_waveforms.png'}")


def plot_unit_templates_figure(waveforms, sorting, plot_path):
    """Plot unit templates."""
    print("Plotting unit templates...")
    plt.figure(figsize=(12, 8))
    plot_unit_templates(
        waveforms,
        unit_ids=sorting.get_unit_ids()[:min(16, len(sorting.get_unit_ids()))],
        ncols=4
    )
    plt.suptitle("Unit Templates", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path / "unit_templates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path / 'unit_templates.png'}")


def plot_raster_figure(sorting, plot_path):
    """Plot raster plot."""
    print("Plotting raster plot...")
    plt.figure(figsize=(15, 8))
    plot_rasters(
        sorting,
        time_range=(0, min(60, sorting.get_total_duration())),
        unit_ids=sorting.get_unit_ids()[:min(20, len(sorting.get_unit_ids()))]
    )
    plt.suptitle("Spike Raster Plot", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path / "raster_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path / 'raster_plot.png'}")


def plot_quality_metrics_distributions(quality_df, plot_path):
    """Plot quality metrics distributions."""
    print("Plotting quality metrics distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # SNR distribution
    snr_data = quality_df['snr'].dropna()
    if len(snr_data) > 0:
        axes[0, 0].hist(snr_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('SNR')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('SNR Distribution')
        axes[0, 0].axvline(snr_data.mean(), color='red', linestyle='--', label=f'Mean: {snr_data.mean():.2f}')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No SNR data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('SNR Distribution')

    # ISI violations
    isi_data = quality_df['isi_violations_ratio'].dropna()
    if len(isi_data) > 0:
        axes[0, 1].hist(isi_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('ISI Violations Ratio')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('ISI Violations Distribution')
        axes[0, 1].axvline(0.1, color='red', linestyle='--', label='Threshold: 0.1')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No ISI data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('ISI Violations Distribution')

    # Presence ratio
    if 'presence_ratio' in quality_df.columns:
        presence_data = quality_df['presence_ratio'].dropna()
        if len(presence_data) > 0:
            axes[1, 0].hist(presence_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Presence Ratio')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Presence Ratio Distribution')
            axes[1, 0].axvline(0.8, color='red', linestyle='--', label='Threshold: 0.8')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No presence ratio data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Presence Ratio Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'Presence ratio not available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Presence Ratio Distribution')

    # Firing rate
    firing_data = quality_df['firing_rate'].dropna()
    if len(firing_data) > 0:
        axes[1, 1].hist(firing_data, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_xlabel('Firing Rate (Hz)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Firing Rate Distribution')
        axes[1, 1].axvline(firing_data.mean(), color='red', linestyle='--', label=f'Mean: {firing_data.mean():.2f} Hz')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No firing rate data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Firing Rate Distribution')

    plt.tight_layout()
    plt.savefig(plot_path / "quality_metrics_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path / 'quality_metrics_distributions.png'}")


def plot_unit_locations_figure(waveforms, sorting, plot_path):
    """Plot unit locations on probe."""
    print("Plotting unit locations on probe...")
    try:
        plt.figure(figsize=(8, 12))
        plot_unit_locations(
            waveforms,
            unit_ids=sorting.get_unit_ids(),
            with_channel_ids=False
        )
        plt.title("Unit Locations on Probe", fontsize=16)
        plt.tight_layout()
        plt.savefig(plot_path / "unit_locations.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path / 'unit_locations.png'}")
    except Exception as e:
        # Fallback: create a simple scatter plot of unit locations
        print(f"Could not use plot_unit_locations, creating simple scatter plot: {e}")
        plt.figure(figsize=(8, 12))
        
        # Get channel locations from waveforms
        channel_locations = waveforms.get_channel_locations()
        unit_ids = sorting.get_unit_ids()
        
        # For each unit, find its peak channel and plot it
        for unit_id in unit_ids[:min(50, len(unit_ids))]:  # Limit to first 50 units
            try:
                template = waveforms.get_template(unit_id)
                peak_channel_idx = np.argmax(np.abs(template))
                if peak_channel_idx < len(channel_locations):
                    x, y = channel_locations[peak_channel_idx]
                    plt.scatter(x, y, c='red', s=20, alpha=0.7)
            except:
                continue
        
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.title("Unit Locations on Probe (Simplified)")
        plt.tight_layout()
        plt.savefig(plot_path / "unit_locations.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path / 'unit_locations.png'}")


def plot_autocorrelograms_figure(sorting, plot_path):
    """Plot autocorrelograms for first few units."""
    print("Plotting autocorrelograms...")
    n_units_to_plot = min(8, len(sorting.get_unit_ids()))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, unit_id in enumerate(sorting.get_unit_ids()[:n_units_to_plot]):
        plot_autocorrelograms(
            sorting,
            unit_ids=[unit_id],
            axes=axes[i],
            bin_ms=1.0,
            window_ms=50.0
        )
        axes[i].set_title(f'Unit {unit_id}')

    plt.suptitle("Autocorrelograms", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path / "autocorrelograms.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path / 'autocorrelograms.png'}")


def print_detailed_results_table(quality_df):
    """Print detailed results table."""
    print("\n=== DETAILED RESULTS TABLE ===")
    print("Unit ID | SNR | ISI Violations | Presence Ratio | Firing Rate | Num Spikes")
    print("-" * 70)

    # Check which columns are available
    available_columns = quality_df.columns.tolist()
    
    for _, row in quality_df.iterrows():
        try:
            # Handle different possible column names
            if 'unit_id' in available_columns:
                unit_id = row['unit_id']
            elif 'unit' in available_columns:
                unit_id = row['unit']
            else:
                unit_id = "N/A"
            
            snr = row.get('snr', float('nan'))
            isi_viol = row.get('isi_violations_ratio', float('nan'))
            presence = row.get('presence_ratio', float('nan'))
            firing_rate = row.get('firing_rate', float('nan'))
            num_spikes = row.get('num_spikes', 0)
            
            # Format the output, handling NaN values
            snr_str = f"{snr:5.2f}" if not pd.isna(snr) else "  NaN"
            isi_str = f"{isi_viol:13.3f}" if not pd.isna(isi_viol) else "          NaN"
            presence_str = f"{presence:13.3f}" if not pd.isna(presence) else "          NaN"
            firing_str = f"{firing_rate:10.2f}" if not pd.isna(firing_rate) else "      NaN"
            
            # Handle num_spikes as either int or float
            try:
                spikes_str = f"{int(num_spikes):9d}"
            except (ValueError, TypeError):
                spikes_str = f"{num_spikes:9.0f}" if not pd.isna(num_spikes) else "      NaN"
            
            print(f"{unit_id:7} | {snr_str} | {isi_str} | {presence_str} | {firing_str} | {spikes_str}")
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue


def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(
        description="View SpikeInterface sorting results from a results folder"
    )
    parser.add_argument(
        "results_path",
        help="Path to the sorting results folder"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots, only print statistics"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively (in addition to saving them)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load results
        sorting, waveforms, quality_df = load_sorting_results(args.results_path)
        
        # Create plots directory
        plot_path = create_plots_directory(Path(args.results_path))
        
        # Print summary statistics
        print_summary_statistics(sorting, quality_df)
        
        if not args.no_plots:
            # Generate all plots with error handling
            try:
                plot_unit_waveforms_figure(waveforms, sorting, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate unit waveforms plot: {e}")
            
            try:
                plot_unit_templates_figure(waveforms, sorting, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate unit templates plot: {e}")
            
            try:
                plot_raster_figure(sorting, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate raster plot: {e}")
            
            try:
                plot_quality_metrics_distributions(quality_df, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate quality metrics distributions: {e}")
            
            try:
                plot_unit_locations_figure(waveforms, sorting, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate unit locations plot: {e}")
            
            try:
                plot_autocorrelograms_figure(sorting, plot_path)
            except Exception as e:
                print(f"Warning: Could not generate autocorrelograms: {e}")
            
            if args.show_plots:
                # Show plots interactively
                plt.show()
        
        # Print detailed results table
        print_detailed_results_table(quality_df)
        
        # Print file locations
        print(f"\n=== FILES SAVED ===")
        print(f"Quality metrics: {Path(args.results_path) / 'postprocessing' / 'quality_metrics.csv'}")
        print(f"Sorting results: {Path(args.results_path) / 'sorting'}")
        print(f"Waveforms: {Path(args.results_path) / 'postprocessing'}")
        if not args.no_plots:
            print(f"Plots: {plot_path}")
            print(f"\nAll visualization plots have been saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
