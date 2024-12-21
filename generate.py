import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif'] = 'SimHei' 

def generate_graphs(n_nodes, p_edge=0.3, is_isomorphic=True):
    # 随机生成G1 可改变策略，如必须连通图
    G1 = nx.erdos_renyi_graph(n_nodes, p_edge)
    """""
    while True:
        G1 = nx.erdos_renyi_graph(n_nodes, p_edge)
        if nx.is_connected(G1):
            break
    """""
    G2 = G1.copy()
    if not is_isomorphic:
        k = 1 # 修改k对边
        n_changes = np.random.randint(1, k+1)  
        for _ in range(n_changes):
            existing_edges = list(G2.edges())
            edge_to_remove = random.choice(existing_edges)
            G2.remove_edge(*edge_to_remove)                     
            while True:
                u, v = random.sample(range(n_nodes), 2)
                if not G2.has_edge(u, v):
                    G2.add_edge(u, v)
                    break 
    # 对G2进行节点重排列，加强随机性
    perm = np.random.permutation(n_nodes)
    adj_matrix = nx.adjacency_matrix(G2).todense()
    adj_matrix_permuted = adj_matrix[perm][:, perm]
    G2 = nx.from_numpy_array(adj_matrix_permuted)  
    return G1, G2

def plot_graphs_and_matrices(G1, G2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  
    # 绘制图
    nx.draw(G1, ax=axes[0,0], with_labels=True, node_color='lightblue')
    axes[0,0].set_title('图1')  
    nx.draw(G2, ax=axes[0,1], with_labels=True, node_color='lightgreen')
    axes[0,1].set_title('图2')
    # 显示邻接矩阵
    axes[1,0].imshow(nx.adjacency_matrix(G1).todense(), cmap='Blues')
    axes[1,0].set_title('图1的邻接矩阵')    
    axes[1,1].imshow(nx.adjacency_matrix(G2).todense(), cmap='Greens')
    axes[1,1].set_title('图2的邻接矩阵')   
    plt.tight_layout()
    plt.show()

# 生成具有8个节点的同构图
G1, G2 = generate_graphs(8, p_edge=0.3, is_isomorphic=True)
plot_graphs_and_matrices(G1, G2)

# 生成具有8个节点的相似图
G1, G2 = generate_graphs(8, p_edge=0.3, is_isomorphic=False)
plot_graphs_and_matrices(G1, G2)