import os
import json
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import random
import warnings
import scipy.sparse.linalg as spla
from laplacian_solver import solve_laplacian_sdd
from advanced_baselines import (
    closeness_energy_scores,
    collective_influence_scores,
    radiation_theory_scores,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 0. Nature 期刊绘图风格高级美学设置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11.5
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

# Nature Publishing Group 配色方案 + 差异化形状标记
STYLE_CONFIG = {
    'STRC': {'color': '#E64B35', 'marker': '*', 'ms': 10, 'z': 10},  # NPG Red, 五角星
    'DC': {'color': '#4DBBD5', 'marker': 'o', 'ms': 5.5, 'z': 5},  # NPG Cyan, 圆形
    'BC': {'color': '#00A087', 'marker': 's', 'ms': 5, 'z': 5},  # NPG Green, 方块
    'CC': {'color': '#F39B7F', 'marker': '^', 'ms': 5.5, 'z': 5},  # NPG Peach, 上三角
    'PR': {'color': '#3C5488', 'marker': 'D', 'ms': 5, 'z': 5},  # NPG Navy, 菱形
    'EC': {'color': '#8491B4', 'marker': 'v', 'ms': 5.5, 'z': 5},  # NPG Purple-Grey, 下三角
    'K-core': {'color': '#7E6148', 'marker': 'p', 'ms': 5.5, 'z': 5},  # NPG Teal, 五边形
    'NC': {'color': '#91D1C2', 'marker': 'X', 'ms': 5.5, 'z': 5},  # Nature Teal/Mint
    'CE': {'color': '#FFB000', 'marker': 'h', 'ms': 5.5, 'z': 5},
    'RT': {'color': '#B07AA1', 'marker': '<', 'ms': 5.5, 'z': 5},
    'CI': {'color': '#59A14F', 'marker': '>', 'ms': 5.5, 'z': 5},

}

def approx_natural_connectivity_centrality(G):
    """自然连通度中心性：节点重要性 = NC(G) - NC(G去掉节点v)
    NC(G) = ln(mean(exp(eigenvalues(A))))
    近似：去掉节点v后 tr(e^A) -= sum_i exp(λ_i)*φ_i(v)^2

    大规模网络优化：
    - N ≤ 500: 完整特征分解（精确）
    - 500 < N ≤ 10000: top-200 特征值
    - N > 10000: top-50 特征值（exp衰减使大特征值主导）
    """
    N = len(G)
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).astype(float)

    if N <= 500:
        eigs, evecs = np.linalg.eigh(A.toarray())
    else:
        # 根据网络规模动态调整特征值数量（平衡精度与速度）
        if N <= 10000:
            k = min(N - 2, 200)
        elif N <= 100000:
            k = min(N - 2, 50)
        else:
            k = min(N - 2, 20)  # 超大网络：仅用 top-20

        eigs, evecs = spla.eigsh(A.tocsr(), k=k, which='LM', tol=1e-3, maxiter=500)

    exp_eigs = np.exp(eigs)
    sum_exp = float(np.sum(exp_eigs))
    nc0 = np.log(sum_exp / N)

    # 各节点对 tr(e^A) 的谱贡献（一阶扰动近似）
    sc = (evecs ** 2) @ exp_eigs  # shape: (N,)

    scores = {}
    for i, u in enumerate(nodes):
        rem = sum_exp - sc[i]
        nc_v = np.log(rem / (N - 1)) if rem > 0 else -np.inf
        scores[u] = float(nc0 - nc_v)
    return scores
# ==========================================
# 1. 核心算法与稳健近似指标库 (与之前保持一致)
# ==========================================
def million_node_stc_approximation(G, K=400):
    N, M = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G).tocsr()
    B = nx.incidence_matrix(G, oriented=True).T
    Q = (np.random.randint(0, 2, size=(K, M)) * 2 - 1) / np.sqrt(K)
    P_T = B.T @ Q.T
    try:
        Z_T, _ = solve_laplacian_sdd(L, P_T, tolerance=1e-5)
    except RuntimeError as exc:
        print(f"[STRC] strict AMG solve failed ({exc}); retrying with W-cycle")
        Z_T, _ = solve_laplacian_sdd(
            L,
            P_T,
            tolerance=1e-5,
            max_cycles=300,
            cycle="W",
            strict=False,
        )
    STC = {}
    for node in nodes:
        i = node_idx[node]
        neighbors = list(G.neighbors(node))
        d_i = len(neighbors)
        if d_i <= 1:
            STC[node] = 0.0;
            continue
        idx_nbr = [node_idx[nbr] for nbr in neighbors]
        Z_diff = Z_T[idx_nbr, :] - Z_T[i, :]
        H_i = Z_diff @ Z_diff.T
        I, J = np.eye(d_i), np.ones((d_i, d_i))
        M_mat = I - H_i + (1.0 / d_i) * J
        c_i = np.linalg.det(M_mat)
        ratio = max(0.0, min(1.0, c_i / d_i))
        STC[node] = 1.0 - ratio
        if STC[node] == 1: STC[node] = d_i
    return STC


def approx_closeness_centrality(G, sample_size=100):
    nodes = list(G.nodes())
    actual_sample = min(len(nodes), sample_size)
    if actual_sample == len(nodes): return nx.closeness_centrality(G)
    samples = random.sample(nodes, actual_sample)
    dist_sum, reachable_count = {n: 0.0 for n in nodes}, {n: 0 for n in nodes}
    for s in samples:
        lengths = nx.single_source_shortest_path_length(G, s)
        for n, d in lengths.items():
            dist_sum[n] += d;
            reachable_count[n] += 1
    closeness = {}
    for n in nodes:
        if dist_sum[n] > 0 and reachable_count[n] > 0:
            closeness[n] = 1.0 / (dist_sum[n] / reachable_count[n])
        else:
            closeness[n] = 0.0
    return closeness


def approx_eigenvector_centrality(G, max_iter=50):
    N = len(G)
    if N == 0: return {}
    A = nx.adjacency_matrix(G).astype(float)
    try:
        _, eigenvector = spla.eigs(A, k=1, which='LM', tol=1e-3, maxiter=200)
        eigenvector = np.abs(eigenvector[:, 0].real)
    except Exception:
        x = np.ones(N) / np.sqrt(N)
        for _ in range(max_iter):
            x_next = A @ x
            norm = np.linalg.norm(x_next)
            if norm == 0: break
            x = x_next / norm
        eigenvector = x
    norm_val = np.linalg.norm(eigenvector)
    if norm_val > 0: eigenvector /= norm_val
    return {list(G.nodes())[i]: float(eigenvector[i]) for i in range(N)}


def get_top_k_nodes(metric_dict, k):
    clean_dict = {n: (v if not np.isnan(v) else 0) for n, v in metric_dict.items()}
    sorted_nodes = sorted(clean_dict.items(), key=lambda x: x[1], reverse=True)
    return [node for node, val in sorted_nodes[:k]]


# ==========================================
# 2. 数据集生成器与 SIR 模型
# ==========================================
def get_exact_connected_er(N=200, M=400):
    while True:
        G = nx.gnm_random_graph(N, M)
        if nx.is_connected(G): return G


def get_exact_ba(N=200, M=400):
    G = nx.barabasi_albert_graph(N, 2)
    while G.number_of_edges() < M:
        u, v = random.sample(list(G.nodes()), 2)
        if not G.has_edge(u, v): G.add_edge(u, v)
    return G


def load_empirical_network(filepath):
    if filepath.endswith('.gml'):
        G = nx.read_gml(filepath)
    else:
        # utf-8-sig transparently strips a UTF-8 BOM while remaining
        # compatible with ordinary UTF-8 edge-list files.
        # Some datasets contain a third timestamp/weight column; we keep only
        # the first two columns to avoid parse errors from nx.read_edgelist.
        edges = []
        with open(filepath, 'r', encoding='utf-8-sig') as edge_file:
            for line in edge_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                edges.append((u, v))
        G = nx.Graph()
        G.add_edges_from(edges)
    G = nx.Graph(G.to_undirected())
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        gcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc_nodes).copy()
    return G


def simulate_sir_final_size(G, seeds, beta, gamma=1.0):
    N = len(G.nodes())
    status = {n: 0 for n in G.nodes()}
    for s in seeds: status[s] = 1
    current_I, recovered_count = set(seeds), 0
    while current_I:
        next_I = set()
        for node in current_I:
            for neighbor in G.neighbors(node):
                if status[neighbor] == 0 and random.random() < beta:
                    status[neighbor] = 1;
                    next_I.add(neighbor)
            if random.random() < gamma:
                status[node] = 2;
                recovered_count += 1
            else:
                next_I.add(node)
        current_I = next_I
    # Every seed is already counted when it enters the recovered state. Adding
    # len(seeds) again would systematically inflate every final outbreak size.
    return recovered_count / N


# ==========================================
# 3. 主干评估逻辑 (带有美学升级的绘图)
# ==========================================
def run_spreading_experiment(synthetic_repeats=10, seed_fraction=0.05):
    datasets = []
    datasets.append(("ER Network", lambda: get_exact_connected_er(200, 400), True))
    datasets.append(("WS Network", lambda: nx.connected_watts_strogatz_graph(200, 4, 0.05), True))
    datasets.append(("BA Network", lambda: get_exact_ba(200, 400), True))

    dataset_dir = SCRIPT_DIR.parent / 'data'
    files = [
        "CA-HepPh.txt",
        "Email-Enron.txt",
        "cit-HepTh.txt",
        "bio-dmela.txt",
        "Wiki-Vote.txt",
        "tech-as-caida.txt",
    ]
    for f in files:
        file_path = os.path.join(dataset_dir, f)
        if os.path.exists(file_path):
            G_real = load_empirical_network(file_path)
            datasets.append((os.path.splitext(f)[0], lambda G=G_real: G.copy(), False))

    seed_percent = 100 * seed_fraction
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
    fig.suptitle(
        f"SIR spreading performance (initial seed fraction = {seed_percent:g}%)",
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    axes_flat = axes.flatten()
    legend_handles = {}

    # 用于存储平均 F 结果（供 visualize_spreading.py 使用）
    f_mean_results = {}

    for i, (net_name, generator_func, is_synthetic) in enumerate(datasets):
        G_sample = generator_func()
        N, E = G_sample.number_of_nodes(), G_sample.number_of_edges()

        current_repeats = synthetic_repeats if is_synthetic else 1
        sir_simulations = 20 if is_synthetic else (5 if N > 10000 else 20)
        print(f"\n[{i + 1}/{len(datasets)}] 评估: {net_name} (N={N}, E={E})")

        degrees = [d for n, d in G_sample.degree()]
        k_avg, k2_avg = np.mean(degrees), np.mean([d ** 2 for d in degrees])
        beta_c = k_avg / k2_avg if k2_avg > 0 else 0.05

        # Normalized infection-rate range used in all SIR experiments.
        # beta / beta^* in {0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5}, clipped to (0, 1].
        beta_multipliers = np.array([0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5])
        betas = np.clip(beta_c * beta_multipliers, np.finfo(float).eps, 1.0)
        if len(betas) == 0 or betas[-1] <= betas[0]:
            betas = np.linspace(np.finfo(float).eps, 1.0, 7)

        results_F_matrix = {name: np.zeros((current_repeats, len(betas))) for name in STYLE_CONFIG.keys()}

        for r in tqdm(range(current_repeats), desc=f"Runs"):
            G = generator_func()
            metrics = {
                'STRC': million_node_stc_approximation(G, K=800),
                'DC': nx.degree_centrality(G),
                'BC': nx.betweenness_centrality(G, k=min(N, 100)),
                'CC': approx_closeness_centrality(G, sample_size=100),
                'PR': nx.pagerank(G),
                'K-core': nx.core_number(G),
                'EC': approx_eigenvector_centrality(G),
                'NC': approx_natural_connectivity_centrality(G),
                'CE': closeness_energy_scores(G),
                'RT': radiation_theory_scores(G),
                'CI': collective_influence_scores(G, radius=2),
            }
            k_num = max(int(N * seed_fraction), 1)
            seeds_dict = {name: get_top_k_nodes(m_dict, k_num) for name, m_dict in metrics.items()}

            for b_idx, beta in enumerate(betas):
                for name, seeds in seeds_dict.items():
                    if name not in STYLE_CONFIG: continue
                    f_vals = [simulate_sir_final_size(G, seeds, beta, 1.0) for _ in range(sir_simulations)]
                    results_F_matrix[name][r, b_idx] = np.mean(f_vals)

        # 绘图区域 ==============================
        ax = axes_flat[i]

        # Nature 风格网格与边框
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#E5E7EB', alpha=0.7, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

        max_F_in_current_subplot = 0.0

        # 计算各方法的平均 F（所有 beta × 所有 repeats 的全局均值）
        dataset_name_map = {
            "CA-HepPh": "CA-HepPh",
            "Email-Enron": "Email-Enron",
            "cit-HepTh": "cit-HepTh",
            "bio-dmela": "bio-dmela",
            "Wiki-Vote": "Wiki-Vote",
            "tech-as-caida": "tech-as-caida",
        }
        display_name = dataset_name_map.get(net_name, net_name)
        f_mean_results[display_name] = [
            float(np.mean(results_F_matrix[name])) for name in STYLE_CONFIG.keys()
        ]

        for name in STYLE_CONFIG.keys():
            # 计算均值和方差（用于绘制误差带）
            mean_F = np.mean(results_F_matrix[name], axis=0)
            std_F = np.std(results_F_matrix[name], axis=0)

            cfg = STYLE_CONFIG[name]
            lw = 2.0 if name == 'STRC' else 1.5
            alpha = 1.0 if name == 'STRC' else 0.85

            # 绘制优雅的半透明标准差阴影区域 (Error Band)
            ax.fill_between(betas, np.clip(mean_F - std_F, 0, 1), np.clip(mean_F + std_F, 0, 1),
                            color=cfg['color'], alpha=0.18 if name == 'STRC' else 0.10, zorder=cfg['z'] - 1)

            # 带白边的折线绘制
            line, = ax.plot(betas, mean_F, color=cfg['color'],
                            linewidth=lw, alpha=alpha, zorder=cfg['z'],
                            marker=cfg['marker'], markersize=cfg['ms'],
                            markeredgewidth=0.8, markeredgecolor='white')

            if name not in legend_handles:
                legend_handles[name] = line

            max_F_in_current_subplot = max(max_F_in_current_subplot, np.max(mean_F + std_F))

        ax.set_title(f"{net_name}\n($N={N}$, $E={E}$)", fontsize=14, pad=8, fontweight='600')

        upper_y = min(max_F_in_current_subplot * 1.1, 1.02) if max_F_in_current_subplot > 0 else 0.1
        ax.set_ylim(0, upper_y)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis='both', which='major', length=4.0, width=1.0, pad=3)

        ax.set_xlabel(r"$\beta$", fontsize=14, labelpad=5)
        ax.set_ylabel(r"$F$", fontsize=14, labelpad=5)

    # 隐藏并布置共用图例
    labels = list(legend_handles.keys())
    handles = [legend_handles[key] for key in labels]

    # 图例
    fig.legend(handles, labels, loc='lower center',
               fontsize=11.5, ncol=6, frameon=False,
               handlelength=2.0, labelspacing=0.8,
               columnspacing=1.2, title='Centrality metric',
               title_fontsize=12)

    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    fig.subplots_adjust(wspace=0.32, hspace=0.45)

    out_base = RESULTS_DIR / 'Network_Spreading_Influence'
    plt.savefig(f'{out_base}.svg', format='svg', dpi=300)
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 将平均 F 结果写入 visualize_spreading.py
    # ==========================================
    _update_visualize_spreading(f_mean_results)


def _update_visualize_spreading(f_mean_results):
    """Store mean outbreak sizes without modifying source files."""
    output_path = RESULTS_DIR / "spreading_results.json"
    output_path.write_text(json.dumps(f_mean_results, indent=2), encoding="utf-8")
    print(f"\n[Spreading results written to {output_path}]")

if __name__ == "__main__":
    run_spreading_experiment(synthetic_repeats=10, seed_fraction=0.05)
