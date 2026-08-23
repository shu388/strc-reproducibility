import numpy as np
import networkx as nx
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from laplacian_solver import solve_laplacian_sdd


# ==========================================
# 核心算法 (R = τ(G-i)/τ(G))
# ==========================================

def get_exact_r(G):
    nodes = list(G.nodes())
    N = len(nodes);
    node_idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G).toarray()
    J_N = np.ones((N, N)) / N
    L_pinv = linalg.inv(L + J_N) - J_N

    r_list = []
    for node in nodes:
        i = node_idx[node];
        nbrs = list(G.neighbors(node));
        d_i = len(nbrs)
        if d_i <= 1: r_list.append(1.0); continue
        idx_nbr = [node_idx[nbr] for nbr in nbrs]
        H_i = L_pinv[np.ix_(idx_nbr, idx_nbr)] - L_pinv[idx_nbr, i][:, None] - L_pinv[idx_nbr, i][None, :] + L_pinv[
            i, i]
        M = np.eye(d_i) - H_i + (1.0 / d_i) * np.ones((d_i, d_i))
        r_list.append(max(0.0, min(1.0, np.linalg.det(M) / d_i)))
    return np.array(r_list)


def get_jl_r(G, K):
    N, M_edges = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G.nodes());
    node_idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G).tocsr()
    B = nx.incidence_matrix(G, oriented=True).T
    Q = (np.random.randint(0, 2, size=(K, M_edges)) * 2 - 1) / np.sqrt(K)
    P_T = B.T @ Q.T
    Z_T, _ = solve_laplacian_sdd(L, P_T, tolerance=1e-5)

    r_list = []
    for node in nodes:
        i = node_idx[node];
        nbrs = list(G.neighbors(node));
        d_i = len(nbrs)
        if d_i <= 1: r_list.append(1.0); continue
        idx_nbr = [node_idx[nbr] for nbr in nbrs]
        Z_diff = Z_T[idx_nbr, :] - Z_T[i, :]
        H_i = Z_diff @ Z_diff.T
        M = np.eye(d_i) - H_i + (1.0 / d_i) * np.ones((d_i, d_i))
        r_list.append(max(0.0, min(1.0, np.linalg.det(M) / d_i)))

    return np.array(r_list)


# ==========================================
# 多场景实验配置
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import spearmanr

# 设置全局字体，符合 Nature 等学术期刊风格
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

def run_multi_network_experiment():
    # 配置实验网络参数
    N = 500
    configs = [
        ("BA Network", nx.barabasi_albert_graph(N, 2), "m=2"),
        ("BA Network", nx.barabasi_albert_graph(N, 5), "m=5"),
        ("ER Network", nx.erdos_renyi_graph(N, 0.015), "p=0.015"),
        ("ER Network", nx.erdos_renyi_graph(N, 0.04), "p=0.04"),
        ("WS Network", nx.watts_strogatz_graph(N, 4, 0.1), "k=4, p=0.1"),
        ("WS Network", nx.watts_strogatz_graph(N, 8, 0.4), "k=8, p=0.4")
    ]

    K = 100
    num_trials = 10

    # 创建子图，Nature 风格尺寸
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), dpi=300)
    axes = axes.flatten()

    # Nature Publishing Group 配色
    COLORS = {
        'primary': '#E64B35',    # NPG Red
        'error': '#F39B7F',      # NPG Peach
        'diagonal': '#7F8C8D',   # Neutral Gray
        'grid': '#E5E7EB'        # Light Gray
    }

    for idx, (title, G, param) in enumerate(configs):
        print(f"正在测试: {title} ({param})...")
        ax = axes[idx]

        # 确保网络连通（取最大连通分量）
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"警告：网络不连通，已提取最大连通子图，剩余节点数: {G.number_of_nodes()}")

        # 计算基准和近似值
        exact_r = get_exact_r(G)          # 需提前定义
        trials = []
        for _ in range(num_trials):
            trials.append(get_jl_r(G, K)) # 需提前定义
        trials = np.array(trials)

        mean_r = np.mean(trials, axis=0)
        std_r = np.std(trials, axis=0)
        corr, _ = spearmanr(exact_r, mean_r)

        # 确定坐标轴范围，包含所有数据点
        min_val = min(np.min(exact_r), np.min(mean_r - std_r))
        max_val = max(np.max(exact_r), np.max(mean_r + std_r))
        margin = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)

        # 隐藏 Top & Right Spines (Nature 期刊边框风格)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # 绘制对角线（柔和灰色虚线代替纯黑）
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--',
                color=COLORS['diagonal'], alpha=0.6, lw=1.0, zorder=1)

        # 绘制误差棒散点图 (NPG 红 + 桃色同色系，干净和谐)
        ax.errorbar(exact_r, mean_r, yerr=std_r,
                    fmt='o', color=COLORS['primary'], ecolor=COLORS['error'],
                    elinewidth=0.7, capsize=1.5, markersize=3.0,
                    markeredgewidth=0, alpha=0.8,
                    label=f'$\\rho = {corr:.4f}$', zorder=2)

        # 设置子图标题和标签
        ax.set_title(f"{title} ({param})", fontsize=14, fontweight='600', pad=8)
        ax.set_xlabel("Exact STRC", fontsize=14, labelpad=4)
        ax.set_ylabel("Approx. STRC", fontsize=14, labelpad=4)
        ax.tick_params(axis='both', which='major', length=4.0, width=1.0, pad=3)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5,
                color=COLORS['grid'], alpha=0.7, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', frameon=True, framealpha=0.92,
                  shadow=False, fontsize=11, edgecolor='#CCCCCC',
                  borderpad=0.4, handletextpad=0.4)

    # 调整子图间距，Nature 风格紧凑布局
    plt.tight_layout(pad=1.2, h_pad=2.5, w_pad=2.0)

    output_dir = Path(__file__).resolve().parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'approximation_accuracy.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'approximation_accuracy.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'approximation_accuracy.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_multi_network_experiment()
