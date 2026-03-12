"""
Convergence Comparison Plot: Latent Space vs Pixel Space
Shows CSI-M vs Epochs and CSI-M vs Wall-Clock Time side by side.

Usage:
    python plot_convergence.py --output convergence_comparison.pdf
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

# ============================================================
# STYLE
# ============================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.dpi': 300,
})

# ============================================================
# DATA — FILL IN YOUR VALUES
# ============================================================

# Validation epochs: 1, 5, 10, 15, 20, 25
val_epochs = [1, 5, 10, 15, 20, 25]

# CSI-M at each validation epoch
csim_latent = [0.2083, 0.3407, 0.3517, 0.3510, 0.3379, 0.3375]   # <-- fill latent space values
csim_pixel  = [0.0800, 0.2998, 0.3164, 0.3191, 0.3187, 0.3179]   # <-- fill pixel space values

# Wall-clock time (in hours) at each validation epoch
# Pull from W&B or training logs
time_latent = [0.25, 1.25, 2.5, 3.75, 5.0, 6.25]   # <-- fill (hours)
time_pixel  = [1.75, 8.75, 17.5, 26.25, 35.0, 43.75]  # <-- fill (hours)

# ============================================================
# COLORS
# ============================================================
c_latent = '#DC143C'   # crimson — matches LASTOCast from leadtime plot
c_pixel  = '#4682B4'   # steel blue — neutral, secondary

# ============================================================
# PLOTTING
# ============================================================

def plot_convergence(output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ----------------------------------------------------------
    # (a) CSI-M vs Epoch
    # ----------------------------------------------------------
    ax1.plot(val_epochs, csim_latent,
             color=c_latent, marker='*', markersize=12, linewidth=2.5,
             label='LASTOCast (latent)', zorder=5,
             markeredgecolor='white', markeredgewidth=0.8)
    ax1.plot(val_epochs, csim_pixel,
             color=c_pixel, marker='o', markersize=8, linewidth=2.0,
             linestyle='--', label='LASTOCast (pixel)', zorder=4,
             markeredgecolor='white', markeredgewidth=0.8)

    # Shade the gap between curves
    ax1.fill_between(val_epochs, csim_latent, csim_pixel,
                     alpha=0.08, color=c_latent)

    ax1.set_xlabel('Epoch', fontweight='medium')
    ax1.set_ylabel('CSI-M ↑', fontweight='medium')
    ax1.set_title('(a) Convergence by Epoch', fontweight='bold', pad=12)
    ax1.set_xticks(val_epochs)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ----------------------------------------------------------
    # (b) CSI-M vs Wall-Clock Time
    # ----------------------------------------------------------
    ax2.plot(time_latent, csim_latent,
             color=c_latent, marker='*', markersize=12, linewidth=2.5,
             label='LASTOCast (latent)', zorder=5,
             markeredgecolor='white', markeredgewidth=0.8)
    ax2.plot(time_pixel, csim_pixel,
             color=c_pixel, marker='o', markersize=8, linewidth=2.0,
             linestyle='--', label='LASTOCast (pixel)', zorder=4,
             markeredgecolor='white', markeredgewidth=0.8)

    # Shade the gap
    # (can't easily fill_between with different x-axes, skip here)

    ax2.set_xlabel('Wall-Clock Time (hours)', fontweight='medium')
    ax2.set_ylabel('')  # shared with left subplot
    ax2.set_title('(b) Convergence by Time', fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Annotate speedup on time plot
    # Uncomment and adjust once you have actual values
    # latent_final_time = time_latent[-1]
    # pixel_final_time = time_pixel[-1]
    # speedup = pixel_final_time / latent_final_time
    # ax2.annotate(f'{speedup:.1f}× faster',
    #              xy=(time_latent[-1], csim_latent[-1]),
    #              xytext=(time_latent[-1] + 1, csim_latent[-1] - 0.03),
    #              fontsize=13, fontweight='bold', color=c_latent,
    #              arrowprops=dict(arrowstyle='->', color=c_latent, lw=1.5),
    #              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                        edgecolor=c_latent, alpha=0.9))

    # ----------------------------------------------------------
    # Shared legend
    # ----------------------------------------------------------
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=2,
               bbox_to_anchor=(0.5, 1.03),
               frameon=True,
               fancybox=True,
               shadow=False,
               edgecolor='#CCCCCC',
               borderpad=0.6,
               columnspacing=2.0,
               handlelength=3.0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='Plots/convergence_comparison.pdf')
    args = parser.parse_args()
    plot_convergence(args.output)
