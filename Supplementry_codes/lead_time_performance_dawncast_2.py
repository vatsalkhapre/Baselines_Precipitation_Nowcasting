import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import os

# ============================================================
# STYLE CONFIGURATION
# ============================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
})

# ============================================================
# DATA — CIKM Values (10 Timesteps)
# ============================================================
timesteps = np.arange(1, 11)  
time_label = "Time Step"

csi = {
    'ConvGRU':              np.array([0.6074, 0.4949, 0.4039, 0.3386, 0.2841, 0.2466, 0.2181, 0.1979, 0.1810, 0.1647]),
    'PhyDNet':              np.array([0.5831, 0.4634, 0.3787, 0.3230, 0.2738, 0.2382, 0.2108, 0.1923, 0.1769, 0.1621]),
    'EarthFormer':          np.array([0.5803, 0.4734, 0.3844, 0.3223, 0.2709, 0.2345, 0.2073, 0.1872, 0.1706, 0.1558]),
    'MAU':                  np.array([0.5816, 0.4693, 0.3814, 0.3188, 0.2711, 0.2443, 0.2152, 0.1953, 0.1791, 0.1652]),
    'Simvp':                np.array([0.5850, 0.4752, 0.3947, 0.3361, 0.2759, 0.2449, 0.2150, 0.1946, 0.1818, 0.1664]),
    'EarthFormer(falfcl)':  np.array([0.5649, 0.4900, 0.4208, 0.3657, 0.3129, 0.2711, 0.2403, 0.2167, 0.1966, 0.1805]),
    'MAU(falfcl)':          np.array([0.5849, 0.4821, 0.4009, 0.3379, 0.2806, 0.2436, 0.2156, 0.1978, 0.1866, 0.1743]),
    'DiffCast':             np.array([0.5758, 0.4543, 0.3628, 0.3036, 0.2572, 0.2305, 0.2090, 0.1908, 0.1757, 0.1616]),
    'AlphaPre':             np.array([0.6001, 0.4831, 0.3894, 0.3112, 0.2563, 0.2145, 0.1886, 0.1683, 0.1510, 0.1389]),
    'DAWN-Cast':            np.array([0.6100, 0.5073, 0.4263, 0.3637, 0.3078, 0.2669, 0.2381, 0.2138, 0.1949, 0.1740]), 
}

hss = {
    'ConvGRU':              np.array([0.7376, 0.6332, 0.5342, 0.4535, 0.3806, 0.3284, 0.2873, 0.2576, 0.2318, 0.2058]),
    'PhyDNet':              np.array([0.7165, 0.6005, 0.5051, 0.4358, 0.3686, 0.3174, 0.2768, 0.2487, 0.2250, 0.2009]),
    'EarthFormer':          np.array([0.7139, 0.6112, 0.5113, 0.4337, 0.3639, 0.3120, 0.2725, 0.2426, 0.2172, 0.1934]),
    'MAU':                  np.array([0.7153, 0.6076, 0.5093, 0.4311, 0.3661, 0.3285, 0.2854, 0.2554, 0.2311, 0.2092]),
    'Simvp':                np.array([0.7176, 0.6129, 0.5240, 0.4519, 0.3705, 0.3252, 0.2817, 0.2507, 0.2311, 0.2059]),
    'EarthFormer(falfcl)':  np.array([0.6991, 0.6275, 0.5534, 0.4883, 0.4196, 0.3609, 0.3159, 0.2803, 0.2488, 0.2228]),
    'MAU(falfcl)':          np.array([0.7176, 0.6194, 0.5309, 0.4542, 0.3765, 0.3247, 0.2841, 0.2574, 0.2403, 0.2210]),
    'DiffCast':             np.array([0.7103, 0.5915, 0.4865, 0.4108, 0.3464, 0.3072, 0.2748, 0.2472, 0.2237, 0.2012]),
    'AlphaPre':             np.array([0.7313, 0.6214, 0.5179, 0.4182, 0.3428, 0.2829, 0.2443, 0.2136, 0.1867, 0.1682]),
    'DAWN-Cast':            np.array([0.7396, 0.6455, 0.5601, 0.4867, 0.4143, 0.3572, 0.3159, 0.2790, 0.2501, 0.2180]),
}

# ============================================================
# MODEL STYLING
# ============================================================
model_styles = {
    'ConvGRU':              {'color': '#7F8C8D', 'marker': 'v', 'ls': '-',  'lw': 1.2, 'ms': 4, 'zorder': 2},
    'PhyDNet':              {'color': '#9B59B6', 'marker': 'D', 'ls': '-',  'lw': 1.2, 'ms': 4, 'zorder': 2},
    'EarthFormer':          {'color': '#2ECC71', 'marker': '^', 'ls': '-',  'lw': 1.2, 'ms': 4, 'zorder': 2},
    'MAU':                  {'color': '#E67E22', 'marker': 'p', 'ls': '-',  'lw': 1.2, 'ms': 4, 'zorder': 2},
    'Simvp':                {'color': '#1ABC9C', 'marker': 'h', 'ls': '-',  'lw': 1.2, 'ms': 4, 'zorder': 2},
    'DiffCast':             {'color': '#E74C3C', 'marker': 's', 'ls': '--', 'lw': 1.5, 'ms': 4, 'zorder': 3},
    'AlphaPre':             {'color': '#3498DB', 'marker': 'o', 'ls': '--', 'lw': 1.5, 'ms': 5, 'zorder': 3},
    'EarthFormer(falfcl)':  {'color': '#27AE60', 'marker': '^', 'ls': ':',  'lw': 1.8, 'ms': 5, 'zorder': 4},
    'MAU(falfcl)':          {'color': '#D35400', 'marker': 'p', 'ls': ':',  'lw': 1.8, 'ms': 5, 'zorder': 4},
    'DAWN-Cast':            {'color': '#C0392B', 'marker': '*', 'ls': '-',  'lw': 2.5, 'ms': 9, 'zorder': 10}, # High Z-order
}

# def plot_leadtime(output_path):
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
#     def draw_metric(ax, data_dict, title, ylabel):
#         for name, values in data_dict.items():
#             if name not in model_styles:
#                 print(f"Warning: {name} not found in model_styles!")
#                 continue
#             s = model_styles[name]
#             ax.plot(timesteps, values,
#                      color=s['color'], marker=s['marker'], markersize=s['ms'],
#                      linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
#                      label=name, markeredgecolor='white', markeredgewidth=0.5)
        
#         ax.set_xlabel(time_label, fontweight='medium')
#         ax.set_ylabel(ylabel, fontweight='medium')
#         ax.set_title(title, fontweight='bold', pad=12)
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     draw_metric(ax1, csi, '(a) CSI vs Lead Time', 'CSI ↑')
#     draw_metric(ax2, hss, '(b) HSS vs Lead Time', 'HSS ↑')
    
#     # --- Shared legend ---
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels,
#                loc='upper center',
#                ncol=5, 
#                bbox_to_anchor=(0.5, 1.05),
#                frameon=True,
#                edgecolor='#CCCCCC',
#                handlelength=3.0)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"Successfully saved plot to: {output_path}")

def plot_csi_single(output_path):
    fig, ax = plt.subplots(figsize=(8, 7)) # Taller aspect ratio for single plot
    
    for name, values in csi.items():
        s = model_styles[name]
        # Use a bold label for the legend
        bold_name = f"$\\bf{{{name}}}$"
        
        ax.plot(timesteps, values,
                 color=s['color'], marker=s['marker'], markersize=s['ms'],
                 linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
                 label=bold_name, markeredgecolor='white', markeredgewidth=0.6)
    
    # Large, bold labels
    ax.set_xlabel(f"$\\bf{{{time_label}}}$", fontweight='bold', labelpad=12, fontsize=18)
    ax.set_ylabel(r"$\bf{CSI \uparrow}$", fontweight='bold', labelpad=12, fontsize=18)
    ax.set_title("Lead-time Performance: CSI", fontweight='bold', pad=25, fontsize=22)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Remove top/right spines for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    bold_labels = [f"$\\bf{{{name}}}$" for name in csi.keys()]
    # Legend: 2 columns to keep the (falfcl) versions side-by-side
    ax.legend(
    loc='upper right', 
    ncol=2, 
    fontsize=16,            # Increases the text size significantly
    prop={'weight': 'bold', 'size': 14}, # Double-ensures boldness and size
    markerscale=1.5,        # Makes the symbols (star, square, etc.) in the legend larger
    handletextpad=0.5,      # Space between the icon and the text
    columnspacing=1.0,      # Space between the two columns
    edgecolor='black',      # Darker border for the legend box to make it "pop"
    framealpha=1.0          # Non-transparent background so lines don't bleed through
)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved optimized single plot to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output', type=str, default='Plots/leadtime_performance.pdf')
    # args = parser.parse_args()
    # plot_leadtime(args.output)
    plot_csi_single('csi_performance_bold.pdf')