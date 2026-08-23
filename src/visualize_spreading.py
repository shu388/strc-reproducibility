import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Define raw data from the spreading simulation results
datasets = ['ER Network', 'WS Network', 'BA Network', 'CA-HepPh', 'Email-Enron', 'cit-HepTh', 'bio-dmela', 'Wiki-Vote', 'tech-as-caida']
methods = ['STRC', 'DC', 'BC', 'CC', 'PR', 'EC', 'K-core', 'NC', 'CE', 'RT', 'CI']
SEED_FRACTION = 0.05

# Raw spreading scale F_mean values (larger is better)
raw_data = {
    "ER Network":    [0.4626, 0.4552, 0.4497, 0.4325, 0.4557, 0.4306, 0.4529, 0.4435, 0.4378, 0.4308, 0.4389],
    "WS Network":    [0.4609, 0.4454, 0.4162, 0.3488, 0.4565, 0.3060, 0.2680, 0.4367, 0.3997, 0.4002, 0.4273],
    "BA Network":    [0.3064, 0.3007, 0.2992, 0.2882, 0.3033, 0.2857, 0.2883, 0.2935, 0.2939, 0.2899, 0.2994],
    "CA-HepPh":    [0.1505, 0.1183, 0.1475, 0.1167, 0.1380, 0.1130, 0.1129, 0.1128, 0.1154, 0.1248, 0.1156],
    "Email-Enron":    [0.1252, 0.1204, 0.1254, 0.1187, 0.1226, 0.1178, 0.1169, 0.1176, 0.1186, 0.1234, 0.1188],
    "cit-HepTh":    [0.1931, 0.1755, 0.1897, 0.1702, 0.1910, 0.1644, 0.1597, 0.1640, 0.1676, 0.1730, 0.1681],
    "bio-dmela":    [0.2261, 0.2077, 0.2119, 0.2034, 0.2130, 0.1983, 0.1982, 0.1985, 0.2023, 0.2076, 0.2039],
    "Wiki-Vote":    [0.1808, 0.1688, 0.1726, 0.1660, 0.1695, 0.1656, 0.1655, 0.1648, 0.1673, 0.1818, 0.1679],
    "tech-as-caida":    [0.1075, 0.1069, 0.1077, 0.1051, 0.1071, 0.1052, 0.1058, 0.1052, 0.1051, 0.1022, 0.1065],
}

# 2. Calculate percentage improvement of STRC compared to other methods
# Since larger F is better, the improvement is: (F_STRC - F_other) / F_other * 100
other_methods = methods[1:]  # Exclude STRC
num_datasets = len(datasets)
num_others = len(other_methods)

improvement_matrix = np.zeros((num_datasets, num_others))

for i, ds in enumerate(datasets):
    f_strc = raw_data[ds][0]
    for j in range(num_others):
        f_other = raw_data[ds][j + 1]
        # Percentage improvement in spreading scale
        improvement_matrix[i, j] = ((f_strc - f_other) / f_other) * 100

# Print calculated values for checking
print("Percentage Improvements (%) of STRC vs Other Methods in Spreading:")
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

# Create the figure with GridSpec and constrained_layout=True
fig = plt.figure(figsize=(16.0, 6.5), dpi=300, constrained_layout=True)
fig.suptitle(
    f"Spreading summary (initial seed fraction = {100 * SEED_FRACTION:g}%)",
    fontsize=16,
    fontweight='bold',
    y=0.99,
)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)

ax_dot = fig.add_subplot(gs[0])
ax_lolly = fig.add_subplot(gs[1], sharey=ax_dot)

# Color palette and markers for centralities (统一颜色配置 - Nature-style NPG colors)
colors = {
    "STRC":    "#E64B35",  # Nature Red
    "DC":      "#4DBBD5",  # NPG Cyan
    "BC":      "#00A087",  # NPG Green
    "CC":      "#F39B7F",  # NPG Peach
    "PR":      "#3C5488",  # NPG Navy
    "EC":      "#8491B4",  # Slate Blue
    "K-core":  "#7E6148",  # NPG Brown
    "NC":      "#91D1C2",  # NPG Light Teal
    "CE":      "#FFB000",
    "RT":      "#B07AA1",
    "CI":      "#59A14F",
}

markers = {
    "STRC":    "*",
    "DC":      "o",
    "BC":      "s",
    "CC":      "^",
    "PR":      "D",
    "EC":      "v",
    "K-core":  "p",
    "NC":      "X",
    "CE":      "h",
    "RT":      "<",
    "CI":      ">",
}

y_positions = np.arange(len(datasets))

# Determine a common x-range that shows all raw F values with a small margin.
all_vals = np.array([raw_data[ds][idx] for ds in datasets for idx in range(len(methods))])
xmin = 0.0
xmax = float(np.max(all_vals) * 1.08)

# --- Panel A: Cleveland Dot Plot of Raw Spreading Scale F ---
for i, ds in enumerate(datasets):
    # Draw horizontal guide lines spanning the full computed range
    ax_dot.hlines(y=i, xmin=xmin, xmax=xmax, colors='#E5E7EB', linestyles='--', linewidth=1.0, zorder=0)

# Plot dot coordinates for each method
for idx, method in enumerate(methods):
    vals = [raw_data[ds][idx] for ds in datasets]
    color = colors[method]
    marker = markers[method]

    if method == "STRC":
        # Highlight STRC with a larger size and black outline
        ax_dot.scatter(vals, y_positions, color=color, marker=marker, s=120,
                       edgecolors='#111111', linewidths=1.0, label=method, zorder=4)
    else:
        ax_dot.scatter(vals, y_positions, color=color, marker=marker, s=55,
                       edgecolors='white', linewidths=0.6, label=method, alpha=0.85, zorder=3)

# Panel A Styling
ax_dot.set_yticks(y_positions)
ax_dot.set_yticklabels(datasets, fontsize=11.5)
ax_dot.set_xlabel("Average Spreading Scale ($F$)", fontsize=14, labelpad=6)
ax_dot.set_xlim(xmin, xmax)
ax_dot.set_title("a  Spreading Performance Comparison (Raw Scale F)", loc='left', fontsize=14, fontweight='600', pad=10)

ax_dot.spines['top'].set_visible(False)
ax_dot.spines['right'].set_visible(False)
ax_dot.spines['left'].set_color('#333333')
ax_dot.spines['bottom'].set_color('#333333')
ax_dot.spines['left'].set_linewidth(1.0)
ax_dot.spines['bottom'].set_linewidth(1.0)

# Legend: clean layout at the bottom, adjusted size and spacing
ax_dot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6, frameon=False, fontsize=10.5)


# --- Panel B: Grouped Horizontal Lollipop Plot of Improvement (%) ---
# Draw vertical baseline at 0% (darker and thicker)
ax_lolly.axvline(0, color='#374151', linestyle='-', linewidth=1.5, zorder=1)

# Position offsets for all baseline methods within each dataset row.
offsets = np.linspace(-0.36, 0.36, num_others)
stem_width = 0.08

for i, ds in enumerate(datasets):
    for j, other in enumerate(other_methods):
        y = i + offsets[j]
        val = improvement_matrix[i, j]
        color = colors[other]
        marker = markers[other]
        
        # Draw the stem (horizontal line from 0 to val, thicker for readability)
        ax_lolly.plot([0, val], [y, y], color=color, linewidth=1.8, alpha=0.9, zorder=2)
        
        # Draw the lollipop head (dot, larger size)
        ax_lolly.scatter(val, y, color=color, marker=marker, s=45, 
                         edgecolors='white', linewidths=0.6, alpha=0.9, zorder=3)

# Panel B Styling
ax_lolly.set_xlabel("Relative Spreading Improvement of STRC (%)", fontsize=14, labelpad=6)
ax_lolly.set_title("b  Spreading Improvement of STRC over Baselines (%)", loc='left', fontsize=14, fontweight='600', pad=10)
ax_lolly.grid(axis='x', linestyle='--', linewidth=0.5, color='#E5E7EB', alpha=0.7, zorder=0)

# Hide y-axis labels for Panel B as they are shared with Panel A
plt.setp(ax_lolly.get_yticklabels(), visible=False)

ax_lolly.spines['top'].set_visible(False)
ax_lolly.spines['right'].set_visible(False)
ax_lolly.spines['left'].set_color('#D1D5DB')  # Soft border between plots
ax_lolly.spines['bottom'].set_color('#333333')
ax_lolly.spines['bottom'].set_linewidth(1.0)

# Tick marks invisible on shared y-axis of panel B
ax_lolly.tick_params(axis='y', which='both', length=0)

# Save the figures
from pathlib import Path
output_dir = Path(__file__).resolve().parent.parent / 'results'
output_dir.mkdir(parents=True, exist_ok=True)
output_png = output_dir / 'STRC_Spreading_Improvement.png'
output_svg = output_dir / 'STRC_Spreading_Improvement.svg'
output_pdf = output_dir / 'STRC_Spreading_Improvement.pdf'
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
print(f"Figures saved to {output_png}, {output_svg} and {output_pdf}")
G_sample = None  # clean memory
