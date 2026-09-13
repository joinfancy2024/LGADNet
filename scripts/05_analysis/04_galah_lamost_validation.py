#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GALAH-LAMOST DR12 cross-match and parameter comparison analysis (streamlined version)

Features:
1. Coordinate cross-matching (1 arcsec radius)
2. Error statistics (MAE, RMSE, R², Bias, Std)
3. Output matched data table
4. Parameter comparison scatter plot (1x4 layout, density-weighted colors)

Output directory: galah_validation/
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root


def log(message: str) -> None:
    """Print a log message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def crossmatch_coordinates(galah_df: pd.DataFrame, lamost_df: pd.DataFrame, radius_arcsec: float = 1.0) -> pd.DataFrame:
    """
    Coordinate cross-matching.

    Args:
        galah_df: GALAH data (ground truth)
        lamost_df: LAMOST prediction data
        radius_arcsec: matching radius (arcsec)

    Returns:
        DataFrame of matched results
    """
    log(f"Start coordinate matching, radius={radius_arcsec} arcsec")

    # Build SkyCoord objects
    galah_coords = SkyCoord(ra=galah_df['RA'].values * u.deg, dec=galah_df['DEC'].values * u.deg, frame='icrs')
    lamost_coords = SkyCoord(ra=lamost_df['ra'].values * u.deg, dec=lamost_df['dec'].values * u.deg, frame='icrs')

    # Cross-match
    log("Performing cross-match (may take a few minutes)...")
    idx_lamost, idx_galah, sep, _ = galah_coords.search_around_sky(lamost_coords, radius_arcsec * u.arcsec)

    log(f"Match result: {len(idx_galah)} pairs")

    if len(idx_galah) == 0:
        log("Warning: no matching sources found! Try increasing the match radius")
        return pd.DataFrame()

    # Filter valid indices
    valid_mask = (idx_galah < len(galah_df)) & (idx_lamost < len(lamost_df))
    idx_galah = idx_galah[valid_mask]
    idx_lamost = idx_lamost[valid_mask]
    sep = sep[valid_mask]

    if len(idx_galah) == 0:
        log("Warning: no valid matching sources after filtering!")
        return pd.DataFrame()

    log(f"Valid matched sources: {len(idx_galah)} pairs")

    # Build the match table
    matched_galah = galah_df.iloc[idx_galah].copy().reset_index(drop=True)
    matched_lamost = lamost_df.iloc[idx_lamost].copy().reset_index(drop=True)

    # Merge data
    matched_df = pd.DataFrame({
        # GALAH data (ground truth)
        'galah_ra': matched_galah['RA'],
        'galah_dec': matched_galah['DEC'],
        'galah_snr': matched_galah['SNR'],
        'galah_logg': matched_galah['LOGG'],
        'galah_teff': matched_galah['TEFF'],
        'galah_c_fe': matched_galah['C_FE'],
        'galah_fe_h': matched_galah['FE_H'],

        # LAMOST data (prediction)
        'lamost_ra': matched_lamost['ra'],
        'lamost_dec': matched_lamost['dec'],
        'lamost_obsid': matched_lamost['obsid'],
        'lamost_logg': matched_lamost['LOGG'],
        'lamost_teff': matched_lamost['TEFF'],
        'lamost_c_fe': matched_lamost['C_FE'],
        'lamost_fe_h': matched_lamost['FE_H'],
        'lamost_class': matched_lamost['class'],
        'lamost_is_cemp': matched_lamost['is_cemp'],

        # Match information
        'separation_arcsec': sep.arcsec,
    })

    # If one GALAH source matches multiple LAMOST observations, keep the closest
    matched_df = matched_df.sort_values(['galah_ra', 'galah_dec', 'separation_arcsec'])
    matched_df = matched_df.drop_duplicates(subset=['galah_ra', 'galah_dec'], keep='first')

    log(f"Matched sources after dedup: {len(matched_df)}")

    return matched_df.reset_index(drop=True)


def calculate_galah_cemp(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the CEMP classification for GALAH data (using a luminosity-adaptive threshold).

    Args:
        matched_df: matched data

    Returns:
        data with the CEMP classification added
    """
    log("Computing GALAH CEMP classification")

    # Compute luminosity
    logg = matched_df['galah_logg']
    teff = matched_df['galah_teff']
    log_l = np.log10(0.8) - (logg - 4.44) + 4.0 * np.log10(teff / 5780.0)

    # CEMP threshold
    threshold = np.where(log_l <= 2.3, 0.7, 3.0 - log_l)

    # Classify
    is_mp = matched_df['galah_fe_h'] < -1.0
    is_cemp = is_mp & (matched_df['galah_c_fe'] >= threshold)

    matched_df['galah_logL'] = log_l
    matched_df['galah_cemp_threshold'] = threshold
    matched_df['galah_is_mp'] = is_mp
    matched_df['galah_is_cemp'] = is_cemp
    matched_df['galah_class'] = 'other'
    matched_df.loc[is_mp & ~is_cemp, 'galah_class'] = 'mp-no-cemp'
    matched_df.loc[is_cemp, 'galah_class'] = 'cemp'

    log(f"GALAH CEMP classification results:\n{matched_df['galah_class'].value_counts()}")

    return matched_df


def plot_comparison(matched_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot the parameter comparison scatter plots (1x4 layout, density-weighted colors).

    Args:
        matched_df: matched data
        output_dir: output directory
    """
    log("Plotting parameter comparison scatter plots")

    # Compute [C/H]
    matched_df['galah_c_h'] = matched_df['galah_c_fe'] + matched_df['galah_fe_h']
    matched_df['lamost_c_h'] = matched_df['lamost_c_fe'] + matched_df['lamost_fe_h']

    # Set plot style
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.dpi'] = 150

    # Parameter list
    params = ['logg', 'teff', 'c_h', 'fe_h']
    param_labels = {
        'logg': r'$\log g$',
        'teff': r'$T_{\mathrm{eff}}$ (K)',
        'c_h': r'[C/H]',
        'fe_h': r'[Fe/H]',
    }

    # Create 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, param in enumerate(params):
        ax = axes[idx]

        galah_col = f'galah_{param}'
        lamost_col = f'lamost_{param}'

        # Extract valid data
        valid = matched_df[[galah_col, lamost_col]].dropna()

        if len(valid) > 0:
            true_vals = valid[galah_col].values
            pred_vals = valid[lamost_col].values

            # Compute errors
            errors = pred_vals - true_vals
            bias = np.mean(errors)
            std = np.std(errors)

            # Compute density (fast estimate using a 2D histogram)
            nbins = 100
            H, xedges, yedges = np.histogram2d(true_vals, pred_vals, bins=nbins)

            # Compute the grid index for each point
            xidx = np.clip(np.digitize(true_vals, xedges) - 1, 0, nbins-1)
            yidx = np.clip(np.digitize(pred_vals, yedges) - 1, 0, nbins-1)

            # Get the density of each point
            density = H[xidx, yidx]

            # Normalize the density
            density_norm = (density - density.min()) / (density.max() - density.min())

            # Order by density (low density drawn first, high density last)
            sorted_idx = np.argsort(density_norm)
            x_sorted = true_vals[sorted_idx]
            y_sorted = pred_vals[sorted_idx]
            density_sorted = density_norm[sorted_idx]

            # Draw the scatter plot, coloring by density
            ax.scatter(x_sorted, y_sorted,
                      c=density_sorted,
                      cmap='coolwarm',  # blue to red
                      s=3,
                      alpha=0.6,
                      edgecolors='none',
                      rasterized=True)

            # 1:1 reference line (black dashed)
            xmin, xmax = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            ax.plot([xmin, xmax], [xmin, xmax], 'k--', linewidth=2, alpha=0.8, label='1:1 line')

            # 3-sigma lines (green dashed)
            y_upper = lambda x: x + bias + 3 * std
            y_lower = lambda x: x + bias - 3 * std

            x_line = np.linspace(xmin, xmax, 100)
            ax.plot(x_line, y_upper(x_line), 'g--', linewidth=1.5, alpha=0.7, label=f'+3σ')
            ax.plot(x_line, y_lower(x_line), 'g--', linewidth=1.5, alpha=0.7, label=f'-3σ')

            # Statistics
            r = pearsonr(true_vals, pred_vals)[0]
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            # Add statistics to the plot
            stats_text = f'MAE = {mae:.3f}\n'
            stats_text += f'RMSE = {rmse:.3f}\n'
            stats_text += f'Bias = {bias:.3f}\n'
            stats_text += f'Std = {std:.3f}\n'
            stats_text += f'R² = {r**2:.3f}'

            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')

            ax.set_xlabel(f"GALAH {param_labels[param]} (True)")
            ax.set_ylabel(f"LAMOST {param_labels[param]} (Predicted)")
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'comparison_scatter.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    log("Parameter comparison scatter plots generated")


def main() -> None:
    """Main function."""
    # Path configuration
    galah_path = LGADNET_ROOT / 'galah_selected_data.csv'
    lamost_path = LGADNET_ROOT / 'predictions/dr12_predict_all.csv'
    output_dir = LGADNET_ROOT / 'galah_validation'

    output_dir.mkdir(parents=True, exist_ok=True)

    log("="*80)
    log("GALAH-LAMOST DR12 cross-match validation analysis (streamlined)")
    log("="*80)

    # Read data (only read required columns to save memory)
    log("Reading GALAH data...")
    galah_df = pd.read_csv(galah_path)
    log(f"GALAH: {len(galah_df)} rows")

    log("Reading LAMOST prediction data (may take a few minutes)...")
    lamost_cols = ['obsid', 'ra', 'dec', 'LOGG', 'TEFF', 'C_FE', 'FE_H', 'class', 'is_cemp']
    lamost_df = pd.read_csv(lamost_path, usecols=lamost_cols)
    log(f"LAMOST: {len(lamost_df)} rows")

    # Cross-match
    matched_df = crossmatch_coordinates(galah_df, lamost_df, radius_arcsec=1.0)

    if len(matched_df) == 0:
        log("Match failed, terminating")
        return

    # Compute GALAH CEMP classification
    matched_df = calculate_galah_cemp(matched_df)

    # Plot parameter comparison scatter plots
    plot_comparison(matched_df, output_dir)

    # Save matched data
    log("Saving matched data...")
    matched_df.to_csv(output_dir / 'galah_matched_comparison.csv', index=False)

    log("="*80)
    log("Validation analysis complete!")
    log(f"Output directory: {output_dir}")
    log(f"Output files:")
    log(f"  - galah_matched_comparison.csv ({len(matched_df)} rows)")
    log(f"  - comparison_scatter.png/pdf (parameter comparison scatter plots)")
    log("="*80)


if __name__ == '__main__':
    main()