import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 1. Define raw data from Table 2
datasets = ['ER Network', 'WS Network', 'BA Network', 'CA-HepPh', 'Email-Enron', 'cit-HepTh', 'bio-dmela', 'Wiki-Vote', 'tech-as-caida']
methods = ['STRC', 'DC', 'BC', 'CC', 'PR', 'EC', 'K-core', 'NC', 'CE', 'RT', 'CI']

# Raw AUC values (R values, smaller is better)
raw_data = {
    "ER Network":    [0.9007, 0.9352, 0.9354, 0.9439, 0.9303, 0.9432, 0.9453, 0.9393, 0.9428, 0.9484, 0.9409],
    "WS Network":    [0.8833, 0.9181, 0.9386, 0.9474, 0.9144, 0.9479, 0.9500, 0.9204, 0.9371, 0.9388, 0.9274],
    "BA Network":    [0.8533, 0.8571, 0.8664, 0.8869, 0.8549, 0.8831, 0.8916, 0.8736, 0.8756, 0.9084, 0.8622],
    "CA-HepPh":    [0.8329, 0.9242, 0.8698, 0.9223, 0.8880, 0.9255, 0.9310, 0.9255, 0.9222, 0.9373, 0.9229],
    "Email-Enron":    [0.4211, 0.4478, 0.4429, 0.5515, 0.4110, 0.5890, 0.5895, 0.5890, 0.5553, 0.9477, 0.4885],
    "cit-HepTh":    [0.9062, 0.9443, 0.9348, 0.9470, 0.9356, 0.9477, 0.9479, 0.9477, 0.9469, 0.9487, 0.9462],
    "bio-dmela":    [0.8083, 0.8113, 0.8127, 0.8518, 0.7883, 0.8691, 0.8740, 0.8692, 0.8497, 0.9191, 0.8303],
    "Wiki-Vote":    [0.7955, 0.8290, 0.7858, 0.8418, 0.8066, 0.8366, 0.8382, 0.8366, 0.8376, 0.9354, 0.8275],
    "tech-as-caida":    [0.1720, 0.0964, 0.1251, 0.4423, 0.0973, 0.4321, 0.2623, 0.4321, 0.4977, 0.9492, 0.1136],
}

# Convert raw data into a numpy array for easy operations (9 datasets x 11 methods).
auc_matrix = np.array([raw_data[ds] for ds in datasets])

# 2. Calculate percentage improvement of STRC compared to other methods
# Since lower AUC is better, the improvement is: (AUC_other - AUC_STRC) / AUC_other * 100
other_methods = methods[1:]  # Exclude STRC
num_datasets = len(datasets)
num_others = len(other_methods)

improvement_matrix = np.zeros((num_datasets, num_others))

for i, ds in enumerate(datasets):
    auc_strc = raw_data[ds][0]
    for j in range(num_others):
        auc_other = raw_data[ds][j + 1]
        # Percentage reduction in AUC (dismantling effectiveness improvement)
        improvement_matrix[i, j] = ((auc_other - auc_strc) / auc_other) * 100

# Print the calculated values for checking
print("Percentage Improvements (%) of STRC vs Other Methods:")
print(f"{'Dataset':<15}", end="")
for m in other_methods:
    print(f"{m:>10}", end="")
print()
print("-" * (15 + 10 * num_others))
for i, ds in enumerate(datasets):
    print(f"{ds:<15}", end="")
    for j in range(num_others):
        print(f"{improvement_matrix[i, j]:>10.2f}%", end="")
    print()
print()

# 3. Setup academic plot parameters (Nature style)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['ytick.major.size'] = 4.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Create the figure with GridSpec and constrained_layout=True to prevent overlaps automatically
fig = plt.figure(figsize=(16.0, 6.0), dpi=300, constrained_layout=True)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], figure=fig)

ax_bar = fig.add_subplot(gs[0])
ax_hm = fig.add_subplot(gs[1])

# --- Panel A: Grouped Bar Chart of Raw AUC ---
# Nature Publishing Group (NPG) color palette (统一颜色配置)
# STRC is colored in distinct Nature Red, while others are in harmonious blue/teal/grey tones
colors = {
    "STRC":    "#E64B35",  # NPG Red (Highlighted)
    "DC":      "#4DBBD5",  # NPG Cyan
    "BC":      "#00A087",  # NPG Green
    "CC":      "#F39B7F",  # NPG Peach
    "PR":      "#3C5488",  # NPG Navy
    "EC":      "#8491B4",  # NPG Slate Blue
    "K-core":  "#7E6148",  # NPG Brown
    "NC":      "#91D1C2",  # NPG Light Teal
    "CE":      "#FFB000",
    "RT":      "#B07AA1",
    "CI":      "#59A14F",
}

x = np.arange(len(datasets))
width = 0.065  # Width of each bar
offset = - (len(methods) - 1) * width / 2

for idx, method in enumerate(methods):
    bar_heights = [raw_data[ds][idx] for ds in datasets]
    color = colors[method]
    
    # Highlight STRC with a bold border
    if method == "STRC":
        rects = ax_bar.bar(x + offset + idx * width, bar_heights, width,
                           label=method, color=color, edgecolor='#111111',
                           linewidth=1.0, zorder=3)
    else:
        rects = ax_bar.bar(x + offset + idx * width, bar_heights, width,
                           label=method, color=color, edgecolor='none',
                           zorder=3)

# Set panel labels & styling
ax_bar.set_ylabel("AUC", fontsize=14, labelpad=6)
ax_bar.set_xticks(x)
# Rotate x-axis labels to avoid overlap and align them nicely
ax_bar.set_xticklabels(datasets, fontsize=11.5, rotation=35, ha='right')
ax_bar.set_ylim(0.0, 1.0)  # Show the full AUC range so all datasets are visible
ax_bar.set_title("a  Dismantling Performance Comparison (Raw AUC)", loc='left', fontsize=14, fontweight='600', pad=10)

# Add grid lines
ax_bar.grid(axis='y', linestyle='--', linewidth=0.5, color='#E5E7EB', alpha=0.7, zorder=0)

# Clean up axes (despine)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_color('#333333')
ax_bar.spines['bottom'].set_color('#333333')
ax_bar.spines['left'].set_linewidth(1.0)
ax_bar.spines['bottom'].set_linewidth(1.0)

# Legend: clean layout at the bottom, adjusted position to avoid overlapping with rotated x-ticks
ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6, frameon=False, fontsize=10.5)


# --- Panel B: Heatmap of Percentage Improvement ---
# Custom diverging/sequential colormap: Yellow-Green-Blue sequential
vmin, vmax = -0.5, 11.0
cmap = plt.colormaps.get_cmap("YlGnBu")

# Plot the heatmap
im = ax_hm.imshow(improvement_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

# Labeling
ax_hm.set_xticks(np.arange(num_others))
ax_hm.set_xticklabels(other_methods, fontsize=11.5)
ax_hm.set_yticks(np.arange(num_datasets))
ax_hm.set_yticklabels(datasets, fontsize=11.5)
ax_hm.set_title("b  AUC Improvement of STRC (%)", loc='left', fontsize=14, fontweight='600', pad=10)

# Annotate each cell with the percentage values
for i in range(num_datasets):
    for j in range(num_others):
        val = improvement_matrix[i, j]
        # Choose text color based on cell brightness (contrasting colors)
        text_color = "white" if val > 6.0 else "black"
        # Prefix with + for positive values to highlight improvement
        text_str = f"{val:+.2f}%" if val != 0 else "0.00%"
        ax_hm.text(j, i, text_str, ha="center", va="center",
                   color=text_color, fontsize=10.5, fontweight='normal')

# Adjust layout and add colorbar
cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=11)
cbar.set_label("Relative AUC Improvement (%)", fontsize=12, labelpad=6)
cbar.outline.set_visible(False)

# Make heatmap frames invisible/clean
for edge, spine in ax_hm.spines.items():
    spine.set_visible(False)

# Set tick markers invisible on heatmap
ax_hm.tick_params(which='both', length=0)

# Save the figures
from pathlib import Path
output_dir = Path(__file__).resolve().parent.parent / 'results'
output_dir.mkdir(parents=True, exist_ok=True)
output_png = output_dir / 'STRC_AUC_Improvement.png'
output_svg = output_dir / 'STRC_AUC_Improvement.svg'
output_pdf = output_dir / 'STRC_AUC_Improvement.pdf'
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
print(f"Figures saved to {output_png}, {output_svg} and {output_pdf}")
