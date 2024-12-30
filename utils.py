import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from scipy.stats import wasserstein_distance
from scipy.stats import entropy


def random_matrix_generator(
    n: int = 50,
    nonzeros_prob: float = 0.1,
    weighted: bool = True,
    normalize: bool = True,
    symmetric: bool = False,
    fill_diagonal: bool = True,
    show_matrix: bool = True,
    save_fig: bool = True,
    save_dir: str = "./image",
) -> np.ndarray:
    """
    Generate a sparse random matrix with n rows and n columns.

    Args:
        n: number of rows and columns.
        nonzeros_prob: probability of non-zero elements.
        weighted: whether to generate a weighted matrix.
        normalize: whether to normalize each row to sum up to 1. i.e., a stochastic matrix.
        symmetric: whether to make the matrix symmetric.
        fill_diagonal: whether to fill the diagonal.
        show_matrix: whether to show the matrix as heatmap.
        save_fig: whether to save the matrix heatmap.
        save_dir: the directory to save the matrix heatmap.
    Returns:
        A random matrix with n rows and n columns.
    """
    if normalize and symmetric:
        print("normalize and symmetric cannot be both True.")
        print(
            "The algorithm will generate a sysmetric matrix first and then normalize it."
        )

    if symmetric:
        S = np.random.rand(n, n)
        # set the upper triangle to zero
        S = np.tril(S)
        S_ = S.T
        S = S + S_ - np.diag(np.diag(S))
    else:
        S = np.random.rand(n, n)
    # make P the indicator matrix
    P = np.zeros((n, n))
    P[S < nonzeros_prob] = 1

    if fill_diagonal:  # whether to guarantee every vertex can go back to itself
        np.fill_diagonal(P, 1)
    else:
        np.fill_diagonal(P, 0)

    # # check if any vertex has no edge
    # for i in range(n):
    #     i_row_sum_without_diagonal = P[i, :].sum() - P[i, i]
    #     i_col_sum_without_diagonal = P[:, i].sum() - P[i, i]
    #     if i_row_sum_without_diagonal < 1e-6 and i_col_sum_without_diagonal < 1e-6:
    #         # randomly choose a number between 0 and n except i
    #         j = i
    #         while j == i:
    #             j = np.random.choice(np.arange(n))
    #         if symmetric:
    #             P[i, j] = 1
    #             P[j, i] = 1
    #         else:
    #             r = np.random.rand()
    #             if r < 0.5:
    #                 P[i, j] = 1
    #             else:
    #                 P[j, i] = 1

    # plot the indicator matrix as heatmap
    # sns.heatmap(P, cmap="YlGnBu", annot=False, fmt=".2f")
    # plt.title("Indicator matrix")
    # plt.show()

    # calculate the degree_in and degree_out of each vertex
    if not symmetric:
        degree_in = np.sum(P, axis=0)
        degree_out = np.sum(P, axis=1)
        # NOTE if degree_out is large, the vertex need more particles to deliver to it
        # NOTE if degree_in is large, many other vertex will deliver to it
        degree_total = degree_in + degree_out - np.diag(P)
    else:
        degree_total = np.sum(P, axis=0)

    if weighted:
        w = degree_total
        W_ = np.tile(w, (n, 1))
        if symmetric:
            W = 0.5 * (W_ + W_.T)
        else:
            W = W_
    else:
        W = np.ones((n, n))

    # if weighted:
    #     W = np.random.rand(n, n)
    #     if symmetric:
    #         W = W + W.T - np.diag(np.diag(W))
    # else:
    #     W = np.ones((n, n))

    # sns.heatmap(P, cmap="YlGnBu", annot=False, fmt=".2f")
    # plt.title("Indicator matrix")
    # plt.show()

    # element-wise multiplication
    A = W * P
    if normalize:
        for i in range(n):
            row_sum = A[i, :].sum()
            if row_sum > 1e-6:
                A[i, :] /= row_sum
            else:
                A[i, :] = np.zeros(n)

    # plot the matrix as heatmap
    if show_matrix:
        info = f"weighted-{weighted}, normalize-{normalize}, symmetric-{symmetric}, fill_diagonal-{fill_diagonal}"
        sns.heatmap(A, cmap="YlGnBu", annot=False, fmt=".2f")
        plt.title(info)
        if save_fig:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"matrix_{info}.png"))
        plt.show()
    return A


def symmetric_matrix_perturbator(
    A, permutation=True, weighted=True, perturbating_prob=0.01
):
    """
    Perturb the given symmetric matrix A with a certain probability.
    If needed, also permute the rows and columns of A.

    Args:
        A: the symmetric matrix to be perturbed.
        permutation: whether to permute the rows and columns of A.
        weighted: whether the A is weighted or not.
        perturbating_prob: the probability of perturbating an element of A.
    Returns:
        The perturbed matrix.
    """
    if not np.allclose(A, A.T):
        print("The matrix is not symmetric.")
        return None

    n = A.shape[0]

    # the indicator matrix, find the elements greater than a very small value
    P = (A > 1e-6).astype(int)

    # calculate the ratio of non-zero elements to all elements
    non_zero_ratio = np.sum(P) / (n**2)

    S = np.random.rand(n, n)

    indices = np.argwhere(S < perturbating_prob)
    # select the indices where i>=j
    for i, j in indices:
        if i < j:
            continue
        r = np.random.rand()
        if r < non_zero_ratio:
            P[i, j] = 1
            P[j, i] = 1
        else:
            P[i, j] = 0
            P[j, i] = 0

    # calculate the degree_in and degree_out of each vertex
    # NOTE it is corresponding to a undirected graph
    degree_total = np.sum(P, axis=0)

    if weighted:
        w = degree_total
        # NOTE if degree_out is large, the vertex need more particles to deliver to it
        # NOTE if degree_in is large, many other vertex will deliver to it
        W_ = np.tile(w, (n, 1))
        W = 0.5 * (W_ + W_.T)
    else:
        W = np.ones((n, n))

    # element-wise multiplication
    A = W * P

    # permute the rows and columns of A
    if permutation:
        perm = np.random.permutation(n)
        A = A[perm][:, perm]

    return A


def eigenvalue_visualizor(
    A,
    title,
    save_fig=True,
    save_dir="./image",
):
    """
    Visualize the eigenvalues of a matrix.

    Args:
        A: a square matrix.
        title: the title of the plot.
        save_fig: whether to save the plot.
        save_dir: the directory to save the plot.
    Returns:
        eigenvalues and eigenvectors of the matrix.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # sort the eigenvalues in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # # visualize the eigenvectors
    # plt.figure(figsize=(10, 5))
    # sns.heatmap(eigenvectors[-10:], cmap="YlGnBu", annot=True, fmt=".2f")
    # plt.title("Eigenvectors")
    # plt.show()

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues, linewidths=0.5, marker="o")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(save_dir, f"eigenvalues_{title}.png"))
    # plt.close()
    plt.show()
    return eigenvalues, eigenvectors


def generate_graphs(
        n_nodes, 
        p_edge=0.2, 
        is_isomorphic=True, 
        is_connected=True,
        is_directed=True,
        k = 1,
    ):
    '''''
    生成一对同构或不同构但相似的图

    输入：
        n_nodes: 节点数
        p_edge: 生成边概率
        is_ismorphic: 是否同构
        is_connected: 是否要求为连通图
        is_directed: 是否为有向图
        k: 若不为同构图，修改k对边
    输出：
        G1, G2两个nx格式图
    '''''
  
    if is_connected:
        while True:
            G1 = nx.erdos_renyi_graph(n_nodes, p_edge, directed=is_directed)
            if is_directed: # 若允许有向图弱连通则改成if False
                if nx.is_strongly_connected(G1):  # 检查是否为强连通图
                    break
            else:
                if nx.is_connected(G1): # 检查是否为连通图
                    break
    else:
        G1 = nx.erdos_renyi_graph(n_nodes, p_edge, directed=is_directed) 

    G2 = G1.copy()
    if not is_isomorphic:
        max_attempts = 100  # 设置最大尝试次数，避免无限循环
        attempt = 0
        
        while attempt < max_attempts:
            # 复制G1作为新的G2
            G2 = G1.copy()
            
            # 获取所有现有边和不存在的边
            existing_edges = list(G2.edges())
            non_existing_edges = [(u, v) for u in range(n_nodes) for v in range(n_nodes) 
                                if u != v and not G2.has_edge(u, v)]
            
            # 确保k不超过可用的边数
            k_actual = min(k, min(len(existing_edges), len(non_existing_edges)))
            
            # 随机选择要删除和添加的边
            edges_to_remove = random.sample(existing_edges, k_actual)
            edges_to_add = random.sample(non_existing_edges, k_actual)
            
            # 执行修改
            for edge in edges_to_remove:
                G2.remove_edge(*edge)
            for edge in edges_to_add:
                G2.add_edge(*edge)
            
            # 检查是否同构
            if not nx.is_isomorphic(G1, G2):
                break
                
            attempt += 1
            
        if attempt == max_attempts:
            raise ValueError(f"生成失败！") 

    # 对G2进行节点重排列，加强随机性
    perm = np.random.permutation(n_nodes)
    adj_matrix = nx.adjacency_matrix(G2).todense()
    adj_matrix_permuted = adj_matrix[perm][:, perm]
    if is_directed:
        G2 = nx.DiGraph(adj_matrix_permuted)  
    else:
        G2 = nx.from_numpy_array(adj_matrix_permuted)  
    return G1, G2

def plot_graphs_and_matrices(G1, G2):
    """''
    展示两张图和对应的邻接矩阵
    
    输入：
        G1, G2: 两个nx格式图
    """ ""
    plt.rcParams["font.sans-serif"] = "SimHei"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 绘制图
    nx.draw(G1, ax=axes[0, 0], with_labels=True, node_color="lightblue")
    axes[0, 0].set_title("图1")
    nx.draw(G2, ax=axes[0, 1], with_labels=True, node_color="lightgreen")
    axes[0, 1].set_title("图2")
    # 显示邻接矩阵
    axes[1, 0].imshow(nx.adjacency_matrix(G1).todense(), cmap="Blues")
    axes[1, 0].set_title("图1的邻接矩阵")
    axes[1, 1].imshow(nx.adjacency_matrix(G2).todense(), cmap="Greens")
    axes[1, 1].set_title("图2的邻接矩阵")
    plt.tight_layout()
    plt.show()


def save_graph(G, name):
    """''
    把图G存储到graphs文件夹的name.pkl
    """ ""
    with open(f"graphs/{name}.pkl", "wb") as f:
        pickle.dump(G, f)


def graph_to_mat(G):
    """''
    将输入的nx格式图转化为numpy格式邻接矩阵
    """ ""
    return nx.adjacency_matrix(G).todense()


def mat_to_graph(mat, is_directed=True):
    """''
    将输入的numpy格式邻接矩阵转化为有向/无向图
    """ ""
    if is_directed:
        G = nx.DiGraph(mat)
    else:
        G = nx.from_numpy_array(mat)
    # nx.draw(G, with_labels=True, node_color='lightblue')
    # plt.show()
    return G


def transition_matrix_iterator(A, x0, iters=100):
    """
    Simulate the Markov chain by matrix multiplication.

    Args:
        A: the transition matrix. Each row of A will be normalized to sum up to 1.
        x0: the initial state.
        iters: the number of iterations. Default is 100.
    Returns:
        A list of states at each iteration.
    """
    x = x0
    states = [x]
    for i in range(iters):
        x = np.dot(A, x)
        states.append(x)

    # plot the states, each line represents a postition in the state vector
    plt.plot(states)
    plt.xlabel("Iteration")
    plt.ylabel("State")
    plt.title("Markov chain simulation")
    plt.show()
    return states


def calculate_mse(distribution1, distribution2):
    """
    计算两个分布之间的归一化均方误差 (MSE) 
    输入：
        distribution1: 第一个分布，字典格式，键为节点，值为频数
        distribution2: 第二个分布，字典格式，键为节点，值为频数
    输出：
        mse: 归一化后的均方误差值
    """
    nodes = distribution1.keys()
    total_samples = sum(distribution1.values())
    mse = np.mean([(distribution1[node] / total_samples - distribution2[node] / total_samples) ** 2 for node in nodes])
    return mse

def js_divergence(p, q):
    """
    计算两个概率分布之间的 Jensen-Shannon 散度
    
    输入：
        p: 第一个概率分布（数组格式）
        q: 第二个概率分布（数组格式）
    输出：
        divergence: JS散度值
    """
    # 确保是概率分布（归一化）
    p = p / np.sum(p)
    q = q / np.sum(q) 
    m = (p + q) / 2
    # JS = 1/2 * (KL(P||M) + KL(Q||M))
    divergence = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return divergence

def simulate_random_walks_until_convergence(G, initial_samples_count, steps, threshold=1e-4, min_steps=10):
    """
    模拟带有收敛条件的随机游走过程
    
    输入：
        G: networkx图对象
        initial_samples_count: 初始节点样本分布，字典格式 {node: count}
        steps: 最大步数限制
        threshold: 收敛阈值，当MSE小于此值时认为收敛
        min_steps: 最小步数要求，避免过早收敛
    
    输出：
        current_samples_count: 最终的样本分布
        steps: 实际执行的步数
    """
    # 初始化样本分布
    prev_samples_count = {node: 0 for node in G.nodes()}
    current_samples_count = initial_samples_count.copy()
    for step in range(1, steps + 1):
        new_samples_count = {node: 0 for node in G.nodes()}     
        # 开始随机游走
        for node in G.nodes():
            for _ in range(current_samples_count[node]):
                current_node = node  
                neighbors = list(G.neighbors(current_node))
                if not neighbors:  
                    break             
                # 计算邻居节点的度数平方并归一化为概率分布
                neighbor_degrees = [G.degree(neighbor)**2 for neighbor in neighbors]
                total_degree = sum(neighbor_degrees)
                probabilities = [degree / total_degree for degree in neighbor_degrees]
                # print(neighbors)
                # print(probabilities)
                # 根据概率分布选择下一个节点
                current_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                new_samples_count[current_node] += 1
        # print(new_samples_count)
        # 计算分布之间的均方误差（MSE）
        mse = calculate_mse(new_samples_count, prev_samples_count)
        # print(f"Step {step}: MSE = {mse}")
        # 检查收敛条件
        if step > min_steps and mse < threshold:
            print(f"随机游走在第 {step} 步收敛 (MSE < {threshold})")
            return new_samples_count, step       
        # 更新分布，进行下一轮步数
        prev_samples_count = current_samples_count.copy()
        current_samples_count = new_samples_count
    print("达到步数限制，停止随机游走")
    return current_samples_count, steps

'''''
# 使用示例
G1, G2 = generate_graphs(20, p_edge=0.1, is_isomorphic=False, is_directed=False, k = 1)
n = 50
initial_samples_count = {node: n for node in G1.nodes()}
steps = 100    # 随机游走的最大步数
threshold = 2e-5  # 收敛条件：归一化 MSE 阈值
# 模拟随机游走
final_samples_count, final_step = simulate_random_walks_until_convergence(G1, initial_samples_count, steps, threshold)
print(f"收敛分布（经过 {final_step} 步）：", final_samples_count)
v1 = np.sort(list(final_samples_count.values()))
final_samples_count, final_step = simulate_random_walks_until_convergence(G2, initial_samples_count, steps, threshold)
print(f"收敛分布（经过 {final_step} 步）：", final_samples_count)
v2 = np.sort(list(final_samples_count.values()))
plot_graphs_and_matrices(G1, G2)
print(v1)
print(v2)
print(nx.is_isomorphic(G1,G2))
mse = np.mean([(v1[i]-v2[i])**2/sum(v1)**2 for i in range(len(v1))])
print(10000 * mse)
dis = wasserstein_distance(v1, v2)
print(dis)
js = js_divergence(v1,v2)
print(1000 * js)
'''''
