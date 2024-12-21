import networkx as nx
import random
import pickle
import matplotlib.pyplot as plt

# 无向图
G = nx.Graph()


'''''
num_nodes = 10
num_edges = 20

# 添加节点
for i in range(1, num_nodes + 1):
    G.add_node(i)

random.seed(12345)

# 添加边
for _ in range(num_edges):
    u = random.randint(1, num_nodes)
    v = random.randint(1, num_nodes)
    if u != v:
        G.add_edge(u, v)

'''''

G.add_nodes_from(range(1, 9))
edges_in_cycle = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
G.add_edges_from(edges_in_cycle)
G.add_edge(7, 1)
G.add_edge(8, 4)

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)
plt.show()

with open('circle14.pkl', 'wb') as f:
    pickle.dump(G, f)