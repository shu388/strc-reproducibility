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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from laplacian_solver import solve_laplacian_sdd
from advanced_baselines import (
    closeness_energy_scores,
    collective_influence_scores,
    radiation_theory_scores,
)

warnings.filterwarnings("ignore")

# 获取脚本所在目录的绝对路径
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
    'K-core': {'color': '#7E6148', 'marker': 'p', 'ms': 5.5, 'z': 5}, # NPG Teal, 五边形
    'NC': {'color': '#91D1C2', 'marker': 'X', 'ms': 5.5, 'z': 5},  # Nature Teal/Mint
    'CE': {'color': '#FFB000', 'marker': 'h', 'ms': 5.5, 'z': 5},
    'RT': {'color': '#B07AA1', 'marker': '<', 'ms': 5.5, 'z': 5},
    'CI': {'color': '#59A14F', 'marker': '>', 'ms': 5.5, 'z': 5},
}


# ==========================================
# 1. 核心算法与稳健近似指标库
# ==========================================

def million_node_stc_approximation(G, K=800):
    N, M = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    L = nx.laplacian_matrix(G).tocsr()
    B = nx.incidence_matrix(G, oriented=True).T
    Q = (np.random.randint(0, 2, size=(K, M)) * 2 - 1) / np.sqrt(K)
    P_T = B.T @ Q.T
    try:
        Z_T, _ = solve_laplacian_sdd(L, P_T, tolerance=1e-5)
    except Exception as exc:
        # Dense, highly clustered graphs can make the default AMG V-cycle
        # stagnate. Retry with a stronger cycle before accepting the finite
        # approximate iterate, so one difficult network does not abort a sweep.
        print(f"[STRC] strict AMG solve failed ({exc}); retrying with W-cycle")
        try:
            Z_T, _ = solve_laplacian_sdd(
                L,
                P_T,
                tolerance=1e-4,
                max_cycles=300,
                cycle="W",
                strict=False,
            )
        except Exception as exc2:
            print(f"[STRC] W-cycle AMG also failed ({exc2}); falling back to scipy CG")
            L_shifted = L + sp.eye(N) * 1e-6
            Z_T = np.zeros((N, K))
            for k in range(K):
                Z_T[:, k], _ = spla.cg(L_shifted, P_T[:, k], tol=1e-4)

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

        I = np.eye(d_i)
        J = np.ones((d_i, d_i))
        M_mat = I - H_i + (1.0 / d_i) * J

        c_i = np.linalg.det(M_mat)
        ratio = max(0.0, min(1.0, c_i / d_i))
        STC[node] = 1.0 - ratio
        if STC[node] == 1: STC[node] = d_i

    return STC

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
def approx_closeness_centrality(G, sample_size=100):
    nodes = list(G.nodes())
    actual_sample = min(len(nodes), sample_size)
    if actual_sample == len(nodes):
        return nx.closeness_centrality(G)

    samples = random.sample(nodes, actual_sample)
    dist_sum, reachable_count = {n: 0.0 for n in nodes}, {n: 0 for n in nodes}

    for s in samples:
        lengths = nx.single_source_shortest_path_length(G, s)
        for n, d in lengths.items():
            dist_sum[n] += d
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
    nodes = list(G.nodes())
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
    return {nodes[i]: float(eigenvector[i]) for i in range(N)}


def simulate_attack(G, scores, top_fraction=0.1):
    N = len(G.nodes())
    num_to_remove = int(N * top_fraction)
    clean_scores = {k: (v if not np.isnan(v) else 0) for k, v in scores.items()}
    ranked_nodes = sorted(clean_scores.keys(), key=lambda x: clean_scores[x], reverse=True)[:num_to_remove]

    G_temp = G.copy()
    s_curve = [len(max(nx.connected_components(G_temp), key=len)) / N] if len(G_temp.nodes()) > 0 else [0.0]

    for node in ranked_nodes:
        G_temp.remove_node(node)
        if len(G_temp.nodes()) > 0:
            gcc_size = len(max(nx.connected_components(G_temp), key=len))
            s_curve.append(gcc_size / N)
        else:
            s_curve.append(0.0)

    return np.array(s_curve)


# ==========================================
# 2. 精确受控生成器与实验框架
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


def load_dataset(filepath):
    if filepath.endswith('.gml'):
        G = nx.read_gml(filepath)
    else:
        # utf-8-sig handles both BOM-prefixed and ordinary UTF-8 edge lists.
        # Some datasets (e.g. sx-mathoverflow) contain a third timestamp column.
        # We only keep the first two columns as endpoints and ignore the rest.
        edges = []
        with open(filepath, 'r', encoding='utf-8-sig') as edge_file:
            for line in edge_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                u, v = parts[0], parts[1]
                edges.append((u, v))
        G = nx.Graph()
        G.add_edges_from(edges)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def run_dismantling_experiment(synthetic_repeats=10, top_fraction=0.1):
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
            G_real = load_dataset(file_path)
            datasets.append((os.path.splitext(f)[0], lambda G=G_real: G.copy(), False))
        else:
            print(f"Warning: Data {f} not found. Please ensure datasets are in {dataset_dir}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
    axes_flat = axes.flatten()
    legend_handles = {}

    # 用于存储 AUC 结果（供 visualize_auc.py 使用）
    auc_results = {}

    for i, (net_name, generator_func, is_synthetic) in enumerate(datasets):
        G_sample = generator_func()
        N, E = G_sample.number_of_nodes(), G_sample.number_of_edges()

        current_repeats = synthetic_repeats if is_synthetic else 1
        print(f"\n[{i + 1}/{len(datasets)}] 正在评估: {net_name} (N={N}, E={E}) | Synthetic: {is_synthetic}")

        num_steps = int(N * top_fraction) + 1
        x_target_percent = np.linspace(0, top_fraction * 100, num_steps)
        all_curves = {name: [] for name in STYLE_CONFIG.keys()}

        for r in tqdm(range(current_repeats), desc=f"Runs for {net_name}"):
            G = generator_func()

            metrics = {
                'STRC': million_node_stc_approximation(G, K=800),  # 名字彻底改为 STRC
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

            for name, scores in metrics.items():
                if name not in STYLE_CONFIG: continue
                curve = simulate_attack(G, scores, top_fraction=top_fraction)

                actual_removed_percent = ((len(curve) - 1) / N) * 100
                x_original_percent = np.linspace(0, actual_removed_percent, len(curve))
                curve_aligned = np.interp(x_target_percent, x_original_percent, curve)
                all_curves[name].append(curve_aligned)

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

        min_curve_value = 1.0

        # 计算 AUC（梯形法则积分，归一化到 [0,1]）
        method_order = list(STYLE_CONFIG)
        auc_dict = {}

        for name in STYLE_CONFIG.keys():
            if not all_curves[name]: continue

            mean_curve = np.mean(all_curves[name], axis=0)
            std_curve = np.std(all_curves[name], axis=0)
            cfg = STYLE_CONFIG[name]

            # 计算 AUC（归一化：面积 / (x_range * y_range)）
            try:
                auc = np.trapezoid(mean_curve, x_target_percent) / (top_fraction * 100)
            except AttributeError:
                # 兼容旧版 NumPy
                auc = np.trapz(mean_curve, x_target_percent) / (top_fraction * 100)
            auc_dict[name] = auc

            lw = 2.0 if name == 'STRC' else 1.5
            alpha = 1.0 if name == 'STRC' else 0.85

            # 绘制优雅的半透明标准差阴影区域 (Error Band)
            ax.fill_between(x_target_percent, np.clip(mean_curve - std_curve, 0, 1), np.clip(mean_curve + std_curve, 0, 1),
                            color=cfg['color'], alpha=0.18 if name == 'STRC' else 0.10, zorder=cfg['z'] - 1)

            # 均匀分布 marker，防止过密
            mark_freq = max(1, len(x_target_percent) // 10)

            # 绘制带有白边的折线
            line, = ax.plot(x_target_percent, mean_curve, color=cfg['color'],
                            linewidth=lw, alpha=alpha, zorder=cfg['z'],
                            marker=cfg['marker'], markersize=cfg['ms'],
                            markevery=mark_freq,
                            markeredgewidth=0.8, markeredgecolor='white')

            if name not in legend_handles:
                legend_handles[name] = line

            min_curve_value = min(min_curve_value, np.min(mean_curve - std_curve))

        # 存储 AUC 结果（按固定顺序输出，匹配 visualize_auc.py 的 methods 列表）
        auc_values = [auc_dict[name] for name in method_order if name in auc_dict]

        # 存储到结果字典（匹配 visualize_auc.py 的数据集命名）
        dataset_name_map = {

            "CA-HepPh": "CA-HepPh",
            "Email-Enron": "Email-Enron",
            "cit-HepTh": "cit-HepTh",
            "bio-dmela": "bio-dmela",
            "Wiki-Vote": "Wiki-Vote",
            "tech-as-caida": "tech-as-caida",
        }
        display_name = dataset_name_map.get(net_name, net_name)
        auc_results[display_name] = auc_values

        ax.set_title(f"{net_name}\n($N={N}$, $E={E}$)", fontsize=14, pad=8, fontweight='600')

        # 智能动态Y轴，稍微给曲线底部留出呼吸空间
        lower_y = max(0.0, min_curve_value - 0.02)
        ax.set_ylim(lower_y, 1.01)

        ax.set_xlim(0, top_fraction * 100)
        ax.set_xticks(np.linspace(0, top_fraction * 100, 6))
        ax.set_xticklabels([f"{int(x)}%" for x in np.linspace(0, top_fraction * 100, 6)])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major', length=4.0, width=1.0, pad=3)

        ax.set_xlabel("$Q$", fontsize=14, labelpad=5)
        ax.set_ylabel("$s(Q)$", fontsize=14, labelpad=5)

    # 布置共用图例
    labels = list(legend_handles.keys())
    handles = [legend_handles[key] for key in labels]
    fig.legend(handles, labels, loc='lower center',
               fontsize=11.5, ncol=6, frameon=False,
               handlelength=2.0, labelspacing=0.8,
               columnspacing=1.2, title='Centrality metric',
               title_fontsize=12)

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.subplots_adjust(wspace=0.32, hspace=0.45)

    out_base = RESULTS_DIR / 'Network_Dismantling'
    plt.savefig(f'{out_base}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 将 AUC 结果写入 visualize_auc.py
    # ==========================================
    _update_visualize_auc(auc_results)


def _update_visualize_auc(auc_results):
    """Store computed AUC values without modifying source files."""
    output_path = RESULTS_DIR / "dismantling_auc.json"
    output_path.write_text(json.dumps(auc_results, indent=2), encoding="utf-8")
    print(f"\n[AUC results written to {output_path}]")


if __name__ == "__main__":
    run_dismantling_experiment(synthetic_repeats=50, top_fraction=0.1)
