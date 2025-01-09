import networkx as nx
from utils import generate_graphs, plot_graphs_and_matrices
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import random
 
def get_node_degrees(G):
    '''
    输入：图G
    输出：图G所有节点的度数
    '''
    # 获取所有节点的度数
    degree_dict = dict(G.degree())   
    # 按度数从大到小排序
    # degree_dict = dict(sorted(degree_dict.items(), key=lambda x: x[1], reverse=True))  
    return degree_dict

def multi_source_bfs_with_depth(G, degree_x):
    '''
    多源带深度BFS
    输入：图G，源节点度数degree_x
    输出：BFS完成后所有节点的度数-深度组合
    '''
    # 获取所有节点的度数
    degree_dict = dict(G.degree())  
    # 找到所有度数为x的节点作为起始点
    start_nodes = [node for node, degree in degree_dict.items() if degree == degree_x]  
    # 初始化结果字典，存储节点的度数和深度
    result = {node: [degree_dict[node], -1] for node in G.nodes()}   
    # 如果没有找到度数为x的节点，直接返回
    if not start_nodes:
        return result  
    # 初始化BFS
    queue = deque()
    visited = set()   
    # 将所有起始节点加入队列，深度设为0
    for node in start_nodes:
        queue.append((node, 0))
        visited.add(node)
        result[node][1] = 0   
    # BFS遍历
    while queue:
        current_node, depth = queue.popleft()       
        # 遍历当前节点的所有邻居
        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                result[neighbor][1] = depth + 1  
    return result


def visualize_graph_with_info(G, result):
    '''
    度数-深度组合可视化
    输入：图G，BFS结果result
    输出：图片
    '''
    plt.rcParams['font.sans-serif'] = 'SimHei' 
    plt.figure(figsize=(12, 8)) 
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500) 
    # 添加节点标签：节点ID\n度数=d\n深度=h
    labels = {node: f"{node}\nd={info[0]}\nh={info[1]}" 
             for node, info in result.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)  
    plt.title("Visualization with degree and depth")
    plt.axis('off')
    plt.show()


'''''
G1, G2 = generate_graphs(500, p_edge=0.05, is_isomorphic=True, is_directed=False, k = 1)

with open('graphs/circle14.pkl', 'rb') as f:
    G1 = pickle.load(f)

with open('graphs/circle13.pkl', 'rb') as f:
    G2 = pickle.load(f)

degrees1 = np.sort(list(get_node_degrees(G1).values()))
degrees2 = np.sort(list(get_node_degrees(G2).values()))
print(degrees1)
print(degrees2)

r1 = multi_source_bfs_with_depth(G1,degrees1[-1])
# print(r1)
visualize_graph_with_info(G1,r1)
sorted_r1 = sorted(list(r1.values()), key=lambda x: (x[0], x[1]))

r2 = multi_source_bfs_with_depth(G2,degrees2[-1])
# print(r2)
visualize_graph_with_info(G2,r2)
sorted_r2 = sorted(list(r2.values()), key=lambda x: (x[0], x[1]))

print(sorted_r1)
print('r2')
print(sorted_r2)
# print(nx.is_isomorphic(G1,G2))
print(sorted_r1 == sorted_r2)
'''''

def running(times):
    '''
    times次数模拟实验，二次BFS验证
    '''
    tp = tn = fp = fn = 0
    for _ in range(times):
        s = random.sample([0,1],1)[0] # 同构or不同构随机
        G1, G2 = generate_graphs(200, p_edge=0.04, is_isomorphic=s, is_directed=False, k = 1)
        degrees1 = sorted(list(get_node_degrees(G1).values()))
        degrees2 = sorted(list(get_node_degrees(G2).values()))
        # print("度数相同？")   
        # print(degrees1 == degrees2)
        r1 = multi_source_bfs_with_depth(G1,degrees1[-1])
        # print(r1)
        # visualize_graph_with_info(G1,r1)
        sorted_r1 = sorted(list(r1.values()), key=lambda x: (x[0], x[1]))

        r2 = multi_source_bfs_with_depth(G2,degrees2[-1])
        # print(r2)
        # visualize_graph_with_info(G1,r2)
        sorted_r2 = sorted(list(r2.values()), key=lambda x: (x[0], x[1]))

        r3 = multi_source_bfs_with_depth(G1,degrees1[1])
        sorted_r3 = sorted(list(r3.values()), key=lambda x: (x[0], x[1]))

        r4 = multi_source_bfs_with_depth(G2,degrees2[1])
        sorted_r4 = sorted(list(r4.values()), key=lambda x: (x[0], x[1]))

        # print('r1')
        # print(sorted_r1)
        # print('r2')
        # print(sorted_r2)
        # print("同构？")
        real_r = nx.is_isomorphic(G1,G2)
        # print(real_r)
        # print("判断结果？")
        classifier1 = sorted_r1 == sorted_r2
        # print(classifier)
        classifier2 = sorted_r3 == sorted_r4
        classifier = classifier1 and classifier2
        if real_r and classifier:
            tp += 1
        elif not classifier and not real_r:
            tn += 1
        elif classifier and not real_r:
            fp += 1
        else:
            fn += 1
        
    return [tp,fp,fn,tn]

x = running(500)
print(x)
        
