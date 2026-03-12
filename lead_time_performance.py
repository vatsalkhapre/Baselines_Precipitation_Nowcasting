"""
Lead-time Performance Plot: CSI-M and HSS vs Time Steps
Generates publication-quality dual subplot figure.

Usage:
    python plot_leadtime.py --output leadtime_performance.pdf
    
Fill in your per-timestep CSI and HSS values below.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

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
# DATA — FILL IN YOUR PER-TIMESTEP VALUES
# ============================================================

# Time steps (minutes) — adjust to match your dataset
# For SEVIR: 5-min intervals, T_out=20 means 5,10,...,100 minutes
# For Shanghai: adjust accordingly
timesteps = np.arange(1, 21)  # or use minutes: np.arange(5, 105, 5)
time_label = "Time Step"       # change to "Time (minutes)" if using minutes

# --- CSI-M per timestep for each model ---
# Each array should have length = number of timesteps (20)
csi = {
    # 'ConvGRU':        np.array([0.0]*20),  # uncomment and fill
    # 'PhyDNet':        np.array([0.0]*20),
    # 'EarthFormer':    np.array([0.0]*20),
    # 'EarthFarseer':   np.array([0.0]*20),
    # 'NowcastNet':     np.array([0.0]*20),
    'DiffCast':       np.array([0.6173920715463559, 0.5110198912894754, 0.44418283646583995, 0.399154072840365, 0.36469394107813563, 0.34124735832895975, 0.32018826166157766, 0.30299726082494727, 0.28792905434141697, 0.274213084221818, 0.2633929401018655, 0.25273032935021317, 0.24318788745507916, 0.23428481803546075, 0.2254796186176284, 0.21813402684479297, 0.2108281069884784, 0.20428806921206044, 0.19815360094825604, 0.19257310221411747]),
    'AlphaPre':       np.array([0.6711275058572944, 0.5712917928047457, 0.4982232696741123, 0.44361959817935515, 0.40236807634467847, 0.36794331597412383, 0.3413084683847119, 0.31964585178675664, 0.3010268612916038, 0.28587603525647814, 0.27306726197075015, 0.26141887948526, 0.2515239489352505, 0.2423188664335044, 0.23364048777759236, 0.22533472174970012, 0.21812705604722407, 0.211130057574292, 0.20406844854917117, 0.1970488928486107]),
    'LASTOCast':      np.array([0.6526194995564145, 0.5826478892255537, 0.5255335592224464, 0.4797784564009146, 0.4415387799667591, 0.41053808669736314, 0.3841413064809876, 0.36202268895298156, 0.3422223225739474, 0.3239880384393596, 0.3082824500163499, 0.2949466946445149, 0.2824888274723574, 0.27026256982377683, 0.2594732150377678, 0.24928227264313862, 0.24014910638022524, 0.23084774112252535, 0.2220174550036925, 0.2146204915527857] ) 
}

# --- HSS per timestep for each model ---
hss = {
    # 'ConvGRU':        np.array([0.0]*20),
    # 'PhyDNet':        np.array([0.0]*20),
    # 'EarthFormer':    np.array([0.0]*20),
    # 'EarthFarseer':   np.array([0.0]*20),
    # 'NowcastNet':     np.array([0.0]*20),
    'DiffCast':       np.array([0.7422552524303274, 0.6419245411395421, 0.5714891484477838, 0.52095099395161, 0.4804185126835476, 0.4521104785310581, 0.4261837356528006, 0.4046640949359348, 0.3855059593508709, 0.3676689146673637, 0.353416995726526, 0.33928928482354476, 0.32652805618847247, 0.31460075457199715, 0.3025602910315518, 0.2924077272691838, 0.2824224674930383, 0.27342102084485803, 0.26516773169233715, 0.2578078502282292]),
    'AlphaPre':       np.array([0.7868659594678088, 0.6979251345377552, 0.6231849733698155, 0.562478056579537, 0.5143410793731842, 0.47239178532018805, 0.4390409200783878, 0.4115585386535401, 0.38743411551402196, 0.36773386591180784, 0.351022265839435, 0.33553518991702447, 0.3225665740329643, 0.3103181720043058, 0.2986416202357163, 0.2871939365059208, 0.27755762688353175, 0.2680340294412089, 0.2585117152356568, 0.24893636857829207]),
    'LASTOCast':      np.array([0.7727016018221106, 0.7116580979430654, 0.6568539537820248, 0.609540682942871, 0.5672823624383754, 0.531709895165341, 0.5002853723469415, 0.4732731313691189, 0.44837222362775747, 0.4247003547351636, 0.40424862875604356, 0.3869954294710074, 0.37053872565475793, 0.35403606660216447, 0.3396028844026094, 0.3257626440776014, 0.31351016762050093, 0.30060457216034786, 0.28852637475279636, 0.27850943166785525]),
}

# ============================================================
# MODEL STYLING — colors, markers, linestyles
# ============================================================

model_styles = {
    'ConvGRU':      {'color': '#8B8000', 'marker': 'v', 'ls': '-',  'lw': 1.5, 'ms': 4, 'zorder': 2},
    'PhyDNet':      {'color': '#6A5ACD', 'marker': 'D', 'ls': '-',  'lw': 1.5, 'ms': 4, 'zorder': 2},
    'EarthFormer':  {'color': '#2E8B57', 'marker': '^', 'ls': '-',  'lw': 1.5, 'ms': 4, 'zorder': 2},
    'EarthFarseer': {'color': '#FF8C00', 'marker': 'p', 'ls': '-',  'lw': 1.5, 'ms': 4, 'zorder': 2},
    'NowcastNet':   {'color': '#20B2AA', 'marker': 'h', 'ls': '-',  'lw': 1.5, 'ms': 4, 'zorder': 2},
    'DiffCast':     {'color': '#CD5C5C', 'marker': 's', 'ls': '--', 'lw': 1.5, 'ms': 4, 'zorder': 2},
    'AlphaPre':     {'color': '#4169E1', 'marker': 'o', 'ls': '--', 'lw': 1.8, 'ms': 5, 'zorder': 3},
    'LASTOCast':    {'color': '#DC143C', 'marker': '*', 'ls': '-',  'lw': 2.5, 'ms': 8, 'zorder': 5},
}

# ============================================================
# PLOTTING
# ============================================================

def plot_leadtime(output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # --- Plot CSI ---
    for name, values in csi.items():
        s = model_styles[name]
        ax1.plot(timesteps, values,
                 color=s['color'], marker=s['marker'], markersize=s['ms'],
                 linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
                 label=name, markeredgecolor='white', markeredgewidth=0.5)
    
    ax1.set_xlabel(time_label, fontweight='medium')
    ax1.set_ylabel('CSI ↑', fontweight='medium')
    ax1.set_title('(a) CSI vs Lead Time', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Plot HSS ---
    for name, values in hss.items():
        s = model_styles[name]
        ax2.plot(timesteps, values,
                 color=s['color'], marker=s['marker'], markersize=s['ms'],
                 linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
                 label=name, markeredgecolor='white', markeredgewidth=0.5)
    
    ax2.set_xlabel(time_label, fontweight='medium')
    ax2.set_ylabel('HSS ↑', fontweight='medium')
    ax2.set_title('(b) HSS vs Lead Time', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # --- Shared legend ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=min(len(csi), 8),
               bbox_to_anchor=(0.5, 1.03),
               frameon=True,
               fancybox=True,
               shadow=False,
               edgecolor='#CCCCCC',
               borderpad=0.6,
               columnspacing=1.2,
               handlelength=2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='Plots/leadtime_performance.pdf')
    args = parser.parse_args()
    plot_leadtime(args.output)