import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

# 无向图
# with open('circle12.pkl', 'rb') as f:
with open('circle14.pkl', 'rb') as f:
    G = pickle.load(f)

# 假设初始时每个节点有 n 个
n = 100
initial_samples_count = {node: n for node in G.nodes()}

def simulate_random_walks(G, initial_samples_count, steps):
    samples_count = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        for _ in range(initial_samples_count[node]):
            current_node = node          
            for _ in range(steps):
                neighbors = list(G.neighbors(current_node))  
                if not neighbors:
                    break  
                current_node = random.choice(neighbors)
            samples_count[current_node] += 1
    return samples_count

# 设定每个随机行走的步数
steps = 50

# 模拟随机游走
final_samples_count = simulate_random_walks(G, initial_samples_count, steps)
print("每个节点上的最终数量:", final_samples_count)

# 绘制图
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)
node_labels = {node: f'\n\n({count})' for node, count in final_samples_count.items()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
plt.show()