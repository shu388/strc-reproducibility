import numpy as np
import networkx as nx
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import time
import warnings
from pathlib import Path
from laplacian_solver import solve_laplacian_sdd

warnings.filterwarnings("ignore")

# Global professional styling parameters (Nature style)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11.5
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['legend.title_fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11.5
plt.rcParams['ytick.labelsize'] = 11.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.major.size'] = 3.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# STRC & JL-STRC Algorithms
# ==========================================

def get_exact_stc(G):
    nodes = list(G.nodes())
    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G).toarray()
    J_N = np.ones((N, N)) / N
    L_pinv = linalg.inv(L + J_N) - J_N
    stc_exact = []
    for node in nodes:
        i = node_idx[node]
        nbrs = list(G.neighbors(node))
        d_i = len(nbrs)
        if d_i <= 1: 
            stc_exact.append(0.0)
            continue
        idx_nbr = [node_idx[nbr] for nbr in nbrs]
        H_i = L_pinv[np.ix_(idx_nbr, idx_nbr)] - L_pinv[idx_nbr, i][:, None] - L_pinv[idx_nbr, i][None, :] + L_pinv[i, i]
        M = np.eye(d_i) - H_i + (1.0 / d_i) * np.ones((d_i, d_i))
        stc_exact.append(1.0 - max(0.0, min(1.0, np.linalg.det(M) / d_i)))
    return np.array(stc_exact)

def get_jl_stc(G, K):
    N, M_edges = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G).tocsr()
    B = nx.incidence_matrix(G, oriented=True).T
    Q = (np.random.randint(0, 2, size=(K, M_edges)) * 2 - 1) / np.sqrt(K)
    P_T = B.T @ Q.T
    Z_T, _ = solve_laplacian_sdd(L, P_T, tolerance=1e-5)
        
    stc_approx = []
    for node in nodes:
        i = node_idx[node]
        nbrs = list(G.neighbors(node))
        d_i = len(nbrs)
        if d_i <= 1: 
            stc_approx.append(0.0)
            continue
        idx_nbr = [node_idx[nbr] for nbr in nbrs]
        Z_diff = Z_T[idx_nbr, :] - Z_T[i, :]
        H_i = Z_diff @ Z_diff.T
        M = np.eye(d_i) - H_i + (1.0 / d_i) * np.ones((d_i, d_i))
        stc_approx.append(1.0 - max(0.0, min(1.0, np.linalg.det(M) / d_i)))
    return np.array(stc_approx)


# ==========================================
# Network Generators (Ensuring Connectedness)
# ==========================================

def get_connected_ba(N, m=4):
    G = nx.barabasi_albert_graph(N, m)
    return G

def get_connected_er(N, avg_deg=8):
    p = avg_deg / N
    while True:
        G = nx.erdos_renyi_graph(N, p)
        if nx.is_connected(G):
            return G
        gcc = max(nx.connected_components(G), key=len)
        if len(gcc) > 0.9 * N:
            return G.subgraph(gcc).copy()

def get_connected_ws(N, k=8, p=0.1):
    return nx.connected_watts_strogatz_graph(N, k, p, tries=100)


# ==========================================
# Main Experiment for Multiple Network Types
# ==========================================

def experiment_multi_network_k_scaling():
    # Use N = 100, 1000, 5000, 10000 to demonstrate clear scaling convergence
    N_list = [100, 1000, 5000, 10000]
    K_values = [20, 50, 100, 200, 400, 600, 800, 1000]
    K_max = max(K_values)
    
    # Configure experiments
    net_types = [
        {"name": "BA Network", "gen": lambda N: get_connected_ba(N, 4)},
        {"name": "ER Network",     "gen": lambda N: get_connected_er(N, 8)},
        {"name": "WS Network", "gen": lambda N: get_connected_ws(N, 8, 0.1)}
    ]

    # Nature Publishing Group palette (npg / lancet inspired).
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), dpi=300, sharey=True, constrained_layout=True)

    for net_idx, net_cfg in enumerate(net_types):
        ax = axes[net_idx]
        net_name = net_cfg["name"]
        print(f"\n========================================")
        print(f"Running Experiment for {net_name} Networks")
        print(f"========================================")
        
        # Setup light grid lines
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#E5E7EB', alpha=0.7, zorder=0)

        for idx, N in enumerate(N_list):
            t_net_start = time.time()
            G = net_cfg["gen"](N)
            
            # 1. Compute exact STRC scores
            t_exact = time.time()
            exact = get_exact_stc(G)
            dt_exact = time.time() - t_exact
            
            # 2. Run JL embedding once for K_max
            M_edges = G.number_of_edges()
            nodes = list(G.nodes())
            N_actual = len(nodes)
            node_idx = {n: i for i, n in enumerate(nodes)}
            L = nx.laplacian_matrix(G).tocsr()
            B = nx.incidence_matrix(G, oriented=True).T
            
            # Generate Q of size K_max x M
            Q = (np.random.randint(0, 2, size=(K_max, M_edges)) * 2 - 1) / np.sqrt(K_max)
            P_T = B.T @ Q.T  # N_actual x K_max

            # Build one AMG hierarchy and reuse it for all K_max right-hand sides.
            t_solve = time.time()
            Z_T, solver_stats = solve_laplacian_sdd(
                L,
                P_T,
                tolerance=1e-5,
            )
            dt_solve = time.time() - t_solve
            
            corrs = []
            print(
                f"  N = {N_actual} (Exact: {dt_exact:.3f}s, "
                f"SDD solve: {dt_solve:.3f}s, "
                f"max residual: {solver_stats.max_relative_residual:.2e})"
            )
            print(f"    {'K':>5} | {'Spearman':>10} | {'Time':>10}")
            print(f"    " + "-" * 32)
            
            for K in K_values:
                t_k_start = time.time()
                # Rescale the prefix of Z_T to correspond to projection dimension K
                Z_T_K = Z_T[:, :K] * np.sqrt(K_max / K)
                
                stc_approx = []
                for node in nodes:
                    i = node_idx[node]
                    nbrs = list(G.neighbors(node))
                    d_i = len(nbrs)
                    if d_i <= 1: 
                        stc_approx.append(0.0)
                        continue
                    idx_nbr = [node_idx[nbr] for nbr in nbrs]
                    Z_diff = Z_T_K[idx_nbr, :] - Z_T_K[i, :]
                    H_i = Z_diff @ Z_diff.T
                    M = np.eye(d_i) - H_i + (1.0 / d_i) * np.ones((d_i, d_i))
                    stc_approx.append(1.0 - max(0.0, min(1.0, np.linalg.det(M) / d_i)))
                
                approx_arr = np.array(stc_approx)
                corr, _ = spearmanr(exact, approx_arr)
                corrs.append(corr)
                dt_k = time.time() - t_k_start
                print(f"    {K:5} | {corr:10.4f} | {dt_k:10.3f}s")
            
            print(f"  Total time for N={N_actual}: {time.time() - t_net_start:.2f}s")
            
            ax.plot(K_values, corrs, marker=markers[idx], label=f'$N={N_actual:,}$',
                    color=colors[idx], markersize=4.5, linewidth=1.5,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=3,
                    clip_on=True)

        # Draw acceptable threshold line
        ax.axhline(y=0.95, color='#6B7280', linestyle=(0, (5, 3)), linewidth=0.9, alpha=0.9, zorder=2)

        # Place the rho=0.95 label below the threshold line on the right, where
        # all curves have plateaued near 1.0, so it is never occluded.
        ax.text(1018, 0.918, r"$\rho=0.95$", color='#4B5563', fontsize=11,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.85))

        # Subplot Titles and Axes Labels
        ax.set_title(f"{net_name}", fontweight='600', pad=6)
        ax.set_xlabel(r'$K$', labelpad=4)

        if net_idx == 0:
            ax.set_ylabel(r"$\rho$", labelpad=4)

        # Tick styling
        ax.tick_params(axis='both', which='major', length=3.5, width=1.0, pad=3)

        # Despine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

        ax.set_ylim(0.3, 1.02)
        ax.set_xlim(-30, 1030)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])

        # Legend (only on the last subplot to avoid duplicate legends)
        if net_idx == 2:
            ax.legend(title="Network size", frameon=True, facecolor='#FFFFFF',
                      edgecolor='#CCCCCC', framealpha=0.98, loc='lower right',
                      borderpad=0.4, handlelength=1.5, handletextpad=0.5,
                      labelspacing=0.35, columnspacing=0.8)

    output_dir = Path(__file__).resolve().parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / 'K_Scaling_Analysis_All.png'
    output_svg = output_dir / 'K_Scaling_Analysis_All.svg'
    output_pdf = output_dir / 'K_Scaling_Analysis_All.pdf'
    plt.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f"\nFigures successfully saved to {output_png} and {output_svg}")
    # plt.show()

if __name__ == "__main__":
    experiment_multi_network_k_scaling()
