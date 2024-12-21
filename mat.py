import numpy as np
adj_matrix = np.zeros((8, 8), dtype=int)
edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
edges.append((7, 1))
edges.append((8, 2))
for i, j in edges:
    adj_matrix[i-1][j-1] = 1  
    adj_matrix[j-1][i-1] = 1
print(adj_matrix)
eigenvalues, _ = np.linalg.eig(adj_matrix)
print(eigenvalues)