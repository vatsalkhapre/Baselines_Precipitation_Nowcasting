"""
Sensitivity Analysis Plot
Generates a figure with 3 subplots showing CSI-M vs each hyperparameter
for Shanghai and MeteoNet.

Usage:
    python plot_sensitivity.py --output sensitivity_analysis.pdf
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2

# ============================================================
# FILL IN YOUR RESULTS HERE
# ============================================================

# Hidden dimension: [16, 32, 64, 128, 256]
dim_values = [16, 32, 64, 128, 256]
dim_shanghai = [0.0, 0.0, 0.0, 0.0, 0.0]      # <-- fill CSI-M values
dim_meteonet = [0.0, 0.0, 0.0, 0.0, 0.0]      # <-- fill CSI-M values
dim_default_idx = 2  # index of default value (64)

# Size factor: [0.25, 0.5, 1.0, 2.0, 4.0]
sf_values = [0.25, 0.5, 1.0, 2.0, 4.0]
sf_shanghai = [0.0, 0.0, 0.0, 0.0, 0.0]       # <-- fill CSI-M values
sf_meteonet = [0.0, 0.0, 0.0, 0.0, 0.0]       # <-- fill CSI-M values
sf_default_idx = 2  # index of default value (1.0)

# Freq multiplier: [0.25, 0.5, 1.0, 1.5, 2.0]
freq_values = [0.25, 0.5, 1.0, 1.5, 2.0]
freq_shanghai = [0.0, 0.0, 0.0, 0.0, 0.0]     # <-- fill CSI-M values
freq_meteonet = [0.0, 0.0, 0.0, 0.0, 0.0]     # <-- fill CSI-M values
freq_default_idx = 2  # index of default value (1.0)

# ============================================================
# PLOTTING
# ============================================================

def plot_sensitivity(output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Colors
    c_shanghai = '#2563EB'   # blue
    c_meteonet = '#DC2626'   # red
    
    # Common styling
    marker_size = 8
    linewidth = 2.0
    
    configs = [
        {
            'ax': axes[0],
            'title': 'Hidden Dimension ($d$)',
            'xlabel': 'Hidden Dimension',
            'values': dim_values,
            'shanghai': dim_shanghai,
            'meteonet': dim_meteonet,
            'default_idx': dim_default_idx,
            'log_scale': True,  # dim values span 16-256, log scale is cleaner
        },
        {
            'ax': axes[1],
            'title': 'MLP Size Factor ($s$)',
            'xlabel': 'Size Factor',
            'values': sf_values,
            'shanghai': sf_shanghai,
            'meteonet': sf_meteonet,
            'default_idx': sf_default_idx,
            'log_scale': True,  # 0.25-4.0, log scale is cleaner
        },
        {
            'ax': axes[2],
            'title': 'Frequency Multiplier ($f$)',
            'xlabel': 'Frequency Multiplier',
            'values': freq_values,
            'shanghai': freq_shanghai,
            'meteonet': freq_meteonet,
            'default_idx': freq_default_idx,
            'log_scale': False,  # 0.25-2.0, linear is fine
        },
    ]
    
    for cfg in configs:
        ax = cfg['ax']
        x = np.arange(len(cfg['values']))
        
        # Plot lines
        ax.plot(x, cfg['shanghai'], color=c_shanghai, marker='o', markersize=marker_size,
                linewidth=linewidth, label='Shanghai', zorder=3)
        ax.plot(x, cfg['meteonet'], color=c_meteonet, marker='s', markersize=marker_size,
                linewidth=linewidth, linestyle='--', label='MeteoNet', zorder=3)
        
        # Highlight default value
        ax.axvline(x=cfg['default_idx'], color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Mark default points with a star
        ax.scatter([cfg['default_idx']], [cfg['shanghai'][cfg['default_idx']]],
                   color=c_shanghai, s=200, marker='*', zorder=5, edgecolors='black', linewidths=0.5)
        ax.scatter([cfg['default_idx']], [cfg['meteonet'][cfg['default_idx']]],
                   color=c_meteonet, s=200, marker='*', zorder=5, edgecolors='black', linewidths=0.5)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in cfg['values']])
        ax.set_xlabel(cfg['xlabel'], fontsize=13, fontweight='medium')
        if cfg['ax'] == axes[0]:
            ax.set_ylabel('CSI-M ↑', fontsize=13, fontweight='medium')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=True)  # keep tick numbers
        ax.set_title(cfg['title'], fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Set y-axis to show meaningful range (auto with small padding)
        all_vals = cfg['shanghai'] + cfg['meteonet']
        non_zero = [v for v in all_vals if v > 0]
        if non_zero:
            ymin = min(non_zero) - 0.02
            ymax = max(non_zero) + 0.02
            ax.set_ylim(ymin, ymax)
    
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True,
               shadow=False, edgecolor='gray')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='sensitivity_analysis.pdf')
    args = parser.parse_args()
    plot_sensitivity(args.output)
