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
# DATA — FILL IN YOUR PER-TIMESTEP VALUES
# ============================================================

# Time steps (minutes) — adjust to match your dataset
# For SEVIR: 5-min intervals, T_out=20 means 5,10,...,100 minutes
# For Shanghai: adjust accordingly
timesteps = np.arange(1, 11)  # or use minutes: np.arange(5, 105, 5)
time_label = "Time Step"       # change to "Time (minutes)" if using minutes

# --- CSI-M per timestep for each model ---
# Each array should have length = number of timesteps (20)
csi = {
    'ConvGRU':        np.array([0.6074017981678002, 0.4949354949693887, 0.40391002539642, 0.3386207488660716, 0.28412218921051735, 0.2466699278946336, 0.21817834555566473, 0.19790048042065805, 0.18101585629542052, 0.16475527198748985]
),  # uncomment and fill
    'PhyDNet':        np.array([0.5831595904843093, 0.4634848554754663, 0.3787714380881697, 0.32305157589579997, 0.27384708433777893, 0.23821218127778976, 0.2108864461037251, 0.192335653929772, 0.1769462556236022, 0.16218964331285907]),
    'EarthFormer':    np.array([0.5803759964419392, 0.4734935776607467, 0.38449921369455287, 0.32234195308001995, 0.27093141604773074, 0.23457074763028865, 0.20733630849543933, 0.18721837136134695, 0.17067994608708298, 0.15581211053121616]),
    'MAU':   np.array([0.5816889739242396, 0.4693631112932155, 0.38143187819846375, 0.3188421343713784, 0.27118931203968794, 0.2443587262893482, 0.215295721136699, 0.19530745535266414, 0.1791470885579374, 0.16523990556782894]),
    'Simvp':     np.array([0.5850657694552881, 0.475238115168978, 0.3947170356219198, 0.33614463262867844, 0.27598173452325686, 0.24491957705120762, 0.2150992090203469, 0.19469374425517552, 0.18183703073795818, 0.1664077520926558]
),
    'EarthFormer(falfcl)': np.array([0.5649617820869963, 0.49003996781618975, 0.4208426930275625, 0.3657216498273408, 0.3129843926163791, 0.27112311906260494, 0.2403551429976535, 0.21674569683714934, 0.1966540416791211, 0.18056464690789667]),
    'MAU(falfcl)': np.array([0.5849499733240578, 0.48215958200683506, 0.40091560539491783, 0.3379556388886284, 0.2806513008252232, 0.24363021687326128, 0.21569405963162303, 0.19784074598161167, 0.1866798258949174, 0.17437785706649578]),
    'DiffCast':       np.array([0.5758531699185823, 0.4543335173940303, 0.36280099339743277, 0.30362726659157563, 0.2572386334082028, 0.23057183044873086, 0.20903138060685683, 0.19083874174114776, 0.17579032372932077, 0.1616981464743448]),
    'AlphaPre':       np.array([0.6001910093228588, 0.4831448998386507, 0.389433246471264, 0.3112833521930525, 0.2563753094695115, 0.21459048422663268, 0.18862082208242953, 0.16830080416999857, 0.15107673227320206, 0.13893256950554111]),
    'DAWN-Cast':      np.array([0.6100359747346659, 0.5073087447566309, 0.4263408659473181, 0.3637228779660465, 0.30785831480706954, 0.2669587542933073, 0.23812485872480266, 0.21380708387984435, 0.19499084864973376, 0.17404966456025783]), 
}

# --- HSS per timestep for each model ---
hss = {
    'ConvGRU':        np.array([0.7376745712974113, 0.6332942213270901, 0.5342009869348944, 0.4535136495648897, 0.38066570384944837, 0.3284454081561779, 0.2873250107055074, 0.2576277865471553, 0.23180273656468664, 0.2058717059755858]
),  # uncomment and fill
    'PhyDNet':        np.array([0.716537617190108, 0.6005503032863592, 0.505132303842304, 0.43587823956506844, 0.3686657448724699, 0.3174184198847668, 0.2768962101343514, 0.2487110673793291, 0.2250222375368341, 0.20099417987806986]),
    'EarthFormer':    np.array([0.7139053384942808, 0.611279939322344, 0.5113776367587262, 0.43372737858343197, 0.36398327860297475, 0.31208272881365795, 0.2725653035347565, 0.24268002078213238, 0.21722856025742399, 0.19344390267768644]),
    'MAU':   np.array([0.71534053768463, 0.6076272579735622, 0.5093467227733877, 0.4311608984319003, 0.3661887813993661, 0.32853274317827735, 0.2854348242130732, 0.2554871928024732, 0.2311357785435709, 0.20927883236092754]
),
    'Simvp':     np.array([0.7176440248418596, 0.6129643767515781, 0.5240912450119714, 0.45191833146791355, 0.3705735056805362, 0.3252355876117544, 0.2817415831892327, 0.2507143668068836, 0.23110716029277398, 0.20592203474205925]
),
    'EarthFormer(falfcl)': np.array([0.699139382268696, 0.6275828565137083, 0.5534950828742284, 0.4883752813918666, 0.41962030231655, 0.3609474364050568, 0.31592894153877954, 0.2803291704852375, 0.2488660724118935, 0.22285390038115752]
),
    'MAU(falfcl)': np.array([0.7176493906658912, 0.6194829847945938, 0.5309759895295192, 0.45421178686212443, 0.3765762876068006, 0.3247545908331613, 0.28414224578006464, 0.25747275204422265, 0.24037969179001725, 0.22101974152655904]),
    'DiffCast':       np.array([0.7103617960517835, 0.5915422143822832, 0.4865706072516047, 0.41087013957306096, 0.34647477160065215, 0.30729894592673423, 0.2748612907942068, 0.2472696947189531, 0.22375608853890788, 0.20126596802816726]),
    'AlphaPre':       np.array([0.7313416495370887, 0.621491750173362, 0.5179628662693634, 0.41823393514604756, 0.3428766095053778, 0.28296518295770434, 0.24437285876324294, 0.21360390565225884, 0.18670430456945705, 0.16822950407160442]),
    'DAWN-Cast':      np.array([0.7396549841549727, 0.6455752330462063, 0.5601350486075924, 0.48675277970203773, 0.41435552003486786, 0.35722824708544426, 0.31598882404674317, 0.2790400762271748, 0.2501689061750398, 0.21807296857444147]),
}

# ============================================================
# MODEL STYLING — colors, markers, linestyles
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
    'DAWN-Cast':    {'color': '#C0392B', 'marker': '*', 'ls': '-',  'lw': 2.5, 'ms': 9, 'zorder': 5},
}

# ============================================================
# PLOTTING
# ============================================================

# def plot_leadtime(output_path):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
#     # --- Plot CSI ---
#     for name, values in csi.items():
#         s = model_styles[name]
#         ax1.plot(timesteps, values,
#                  color=s['color'], marker=s['marker'], markersize=s['ms'],
#                  linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
#                  label=name, markeredgecolor='white', markeredgewidth=0.5)
    
#     ax1.set_xlabel(time_label, fontweight='medium')
#     ax1.set_ylabel('CSI ↑', fontweight='medium')
#     ax1.set_title('(a) CSI vs Lead Time', fontweight='bold', pad=10)
#     ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
    
#     # --- Plot HSS ---
#     for name, values in hss.items():
#         s = model_styles[name]
#         ax2.plot(timesteps, values,
#                  color=s['color'], marker=s['marker'], markersize=s['ms'],
#                  linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
#                  label=name, markeredgecolor='white', markeredgewidth=0.5)
    
#     ax2.set_xlabel(time_label, fontweight='medium')
#     ax2.set_ylabel('HSS ↑', fontweight='medium')
#     ax2.set_title('(b) HSS vs Lead Time', fontweight='bold', pad=10)
#     ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
    
#     # --- Shared legend ---
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels,
#                loc='upper center',
#                ncol=min(len(csi), 8),
#                bbox_to_anchor=(0.5, 1.03),
#                frameon=True,
#                fancybox=True,
#                shadow=False,
#                edgecolor='#CCCCCC',
#                borderpad=0.6,
#                columnspacing=1.2,
#                handlelength=2.5)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"Saved to {output_path}")
#     plt.close()

def plot_leadtime(output_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # helper to plot subplots
    def draw_metric(ax, data_dict, title, ylabel):
        for name, values in data_dict.items():
            if name not in model_styles: continue
            s = model_styles[name]
            ax.plot(timesteps, values,
                     color=s['color'], marker=s['marker'], markersize=s['ms'],
                     linewidth=s['lw'], linestyle=s['ls'], zorder=s['zorder'],
                     label=name, markeredgecolor='white', markeredgewidth=0.5)
        
        ax.set_xlabel(time_label, fontweight='medium')
        ax.set_ylabel(ylabel, fontweight='medium')
        ax.set_title(title, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    draw_metric(ax1, csi, '(a) CSI vs Lead Time', 'CSI ↑')
    draw_metric(ax2, hss, '(b) HSS vs Lead Time', 'HSS ↑')
    
    # --- Shared legend ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=5, # Adjusted for better fit
               bbox_to_anchor=(0.5, 1.05),
               frameon=True,
               edgecolor='#CCCCCC',
               handlelength=3.0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='Plots/leadtime_performance.pdf')
    args = parser.parse_args()
    plot_leadtime(args.output)