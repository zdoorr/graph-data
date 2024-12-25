import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        print("symmetric will be set to False.")
        symmetric = False
    if weighted:
        W = np.random.rand(n, n)
        if symmetric:
            W = W + W.T - np.diag(np.diag(W))
    else:
        W = np.ones((n, n))
    if symmetric:
        P = np.random.rand(n, n)
        # set the upper triangle to zero
        P = np.tril(P)
        P_ = P.T
        P = P + P_ - np.diag(np.diag(P))
    else:
        P = np.random.rand(n, n)
    P[P < nonzeros_prob] = 1
    P[P >= nonzeros_prob] = 0

    if fill_diagonal:
        np.fill_diagonal(P, 1)
    else:
        np.fill_diagonal(P, 0)
    # check if any vertex has no edge
    for i in range(n):
        i_row_sum_without_diagonal = P[i, :].sum() - P[i, i]
        i_col_sum_without_diagonal = P[:, i].sum() - P[i, i]
        if i_row_sum_without_diagonal < 1e-6 and i_col_sum_without_diagonal < 1e-6:
            # randomly choose a number between 0 and n except i
            j = np.random.choice(np.arange(n))
            if j == i:
                j = (i + 1) % n
            if symmetric:
                P[i, j] = 1
                P[j, i] = 1
            else:
                r = np.random.rand()
                if r < 0.5:
                    P[i, j] = 1
                else:
                    P[j, i] = 1

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
        sns.heatmap(A, cmap="YlGnBu", annot=False, fmt=".2f")
        plt.title(info)
        if save_fig:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            info = f"weighted-{weighted}, normalize-{normalize}, symmetric-{symmetric}, fill_diagonal-{fill_diagonal}"
            plt.savefig(os.path.join(save_dir, f"matrix_{info}.png"))
        plt.show()
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

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(save_dir, f"eigenvalues_{title}.png"))
    # plt.close()
    plt.show()
    return eigenvalues, eigenvectors