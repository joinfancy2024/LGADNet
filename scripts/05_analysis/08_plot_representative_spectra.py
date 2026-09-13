#!/usr/bin/env python3
"""
Plot representative spectra comparison - improved version of the original code

Notes on changes:
1. Dataset: lgadnet_dataset
2. Sample selection: for all classes, pick the sample closest to the C_FE median (instead of random selection)
3. Style: fully preserves the original code style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 0. Set random seed (for reproducibility) =================
# Note: now uses median selection, random seed no longer needed
# np.random.seed(42)  # deprecated

# ================= Configure paths =================
LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

x_path = LGADNET_ROOT / "lgadnet_dataset/train_features.npy"
y_path = LGADNET_ROOT / "lgadnet_dataset/train_labels.csv"
save_path = LGADNET_ROOT / "figures/representative_spectra.pdf"

# ================= 1. Load data =================
# Read labels
df_labels = pd.read_csv(y_path)

# Load spectra (use mmap_mode='r' to save memory)
data_spectra = np.load(x_path, mmap_mode='r')

# Check lengths to build the wavelength axis
n_samples, n_points = data_spectra.shape
print(f"Data loaded: {n_samples} spectra, each {n_points} points")

# Build wavelength axis (3900 - 8800)
wavelength = np.linspace(3900, 8800, n_points)

# ================= 2. Select representative samples (median strategy) =================
classes = ['cemp', 'mp-no-cemp', 'other']

samples = {}
for cls in classes:
    # Get all indices for this class
    df_class = df_labels[df_labels['class'] == cls]

    if len(df_class) > 0:
        # Improvement: pick the sample closest to the C_FE median (instead of random selection)
        median_cfe = df_class['C_FE'].median()
        closest_idx = (df_class['C_FE'] - median_cfe).abs().idxmin()

        samples[cls] = data_spectra[closest_idx]

        # Output the selected sample info
        sample_info = df_class.loc[closest_idx]
        print(f"Class {cls}: selected sample index {closest_idx}")
        print(f"  ObsID={int(sample_info['obsid'])}, [C/Fe]={sample_info['C_FE']:.2f}, [Fe/H]={sample_info['FE_H']:.2f}, Teff={sample_info['TEFF']:.0f}K")
    else:
        print(f"Warning: no samples found for class {cls}")

# ================= 3. Define important spectral lines (units: Angstrom) =================
# Line colors: red for carbon lines, blue for iron lines
lines_info = {
    # --- Carbon related (Carbon / CH / C2) ---
    4300: ('CH G-band', 'red'),
    4737: ('C₂ Swan', 'red'),
    5165: ('C₂ Swan', 'red'),
    5635: ('C₂ Swan', 'red'),

    # --- Iron related (Iron / Fe) ---
    4383: ('Fe I', 'blue'),
    5270: ('Fe I', 'blue'),
    5335: ('Fe I', 'blue'),
}

# ================= 4. Plot =================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=100)
plt.subplots_adjust(hspace=0.1)

for i, cls in enumerate(classes):
    ax = axes[i]
    if cls not in samples:
        continue

    flux = samples[cls]

    # Simple normalization (divide by median)
    norm_flux = flux / (np.median(flux) + 1e-6)

    # Plot spectrum (research-paper standard dark blue)
    ax.plot(wavelength, norm_flux, color='#2166AC', linewidth=0.8, label=f'Class: {cls}')

    # Mark important lines
    for wl_val, (name, color) in lines_info.items():
        ax.axvline(x=wl_val, color=color, linestyle='--', alpha=0.5, linewidth=1)

        # Only annotate on the first panel
        if i == 0:
            ax.text(wl_val, ax.get_ylim()[1]*0.95, name, color=color,
                    rotation=90, fontsize=8, ha='right', va='top')

    # G-band region shading (semi-transparent light blue)
    if i == 0:
        ax.axvspan(4280, 4320, color='#85C1E9', alpha=0.3, label='Carbon Features')
    else:
        ax.axvspan(4280, 4320, color='#85C1E9', alpha=0.3)

    # Legend and labels
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylabel('Normalized Flux')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Sample Spectrum: {cls}", fontsize=12)

axes[-1].set_xlabel(r'Wavelength ($\AA$)')
axes[-1].set_xlim(3900, 6000) # Limit the display range

# Remove the big title to keep it clean
# plt.suptitle('Comparison of Spectra Classes with Key C & Fe Lines', fontsize=16)
plt.tight_layout()

# Save figure
plt.savefig(save_path)
print(f"\nFigure saved to: {save_path}")
plt.show()
