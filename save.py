import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

# 有向图
G = nx.DiGraph()
num_nodes = 10
num_edges = 20
for i in range(1, num_nodes + 1):
    G.add_node(i)

random.seed(12345)

for _ in range(num_edges):
    u = random.randint(1, num_nodes)
    v = random.randint(1, num_nodes)
    if u != v:  
        G.add_edge(u, v)

with open('graph.pkl', 'wb') as f:
    pickle.dump(G, f)
