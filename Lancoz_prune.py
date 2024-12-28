"""
This is the code for pruning by Lancoz algorithm.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


def lancoz_pruner(A, target_eigenvalue):
    """
    This function do the Lancoz iteration on a sparse matrix A.
    And calculathe the eigenvalue of the Hessenberg matrix on each iteration.
    Then use the Cauchy interlacing theorem to check if the eigenvalue
    of Hessenberg matrix is within the target_eigenvalue.
    If not, then the eigenvalue of A is not equal to the target_eigenvalue, and return False.
    If yes, then continue the iteration.

    Args:
        A: a sparse matrix
        target_eigenvalue: the target eigenvalue

    Returns:
        result: a boolean value
        n: the number of iterations
    """
    n = A.shape[0]
    alphas = np.zeros(n + 1)
    betas = np.zeros(n)
    Q = np.zeros((n, n + 1))
    q1 = np.random.rand(n)
    q1 = q1 / np.linalg.norm(q1)
    Q[:, 1] = q1
    for i in range(1, n):
        v = A @ Q[:, i]
        alphas[i] = np.dot(Q[:, i], v)
        v = v - betas[i - 1] * Q[:, i - 1] - alphas[i] * Q[:, i]
        betas[i] = np.linalg.norm(v)
        Q[:, i + 1] = v / betas[i]
        T = sp.diags([alphas[1 : i + 1], betas[1:i], betas[1:i]], [0, -1, +1])
        # # plot T
        # sns.heatmap(T.toarray(), cmap="coolwarm")
        # plt.title("T matrix")
        # plt.show()
        # calculate the eigenvalue of T
        eigenvalue = np.linalg.eigvalsh(T.toarray())
        # use Cauchy interlacing theorem to check if the eigenvalue is within the target_eigenvalue
        if (target_eigenvalue[:i] - 1e-3 <= eigenvalue).all() and (
            eigenvalue <= target_eigenvalue[-i:] + 1e-3
        ).all():
            continue
        else:
            # print("not interlacing!")
            # print("Stop at iteration:", i)
            # print(eigenvalue)
            # print(target_eigenvalue)
            # C = Q[:, 1 : (i + 1)].T @ Q[:, 1 : (i + 1)]
            # sns.heatmap(C, cmap="coolwarm")
            # plt.title("Q matrix inner product")
            # plt.show()
            # print(target_eigenvalue[:i] <= eigenvalue)
            # print(eigenvalue <= target_eigenvalue[-i:])
            return False, i
    return True, n


def mean_iter_finder(V, p):
    """
    This function find the mean iterationsof iterations for Lancoz algorithm before i lose othogonalization.

    Args:
        V: the number of vertices
        p: the probability of non-zero elements in the matrix

    Returns:
        mean_iter: the mean number of iterations
    """
    stop_iter_list = []
    for _ in range(10):
        A = random_matrix_generator(
            n=V,
            nonzeros_prob=p,
            weighted=True,
            normalize=False,
            symmetric=True,
            fill_diagonal=False,
            show_matrix=False,
            save_fig=False,
        )
        target_eigenvalue = np.sort(np.linalg.eigvalsh(A))

        for _ in range(10):
            # permute the rows and columns of A
            perm = np.random.permutation(V)
            A_perm = A[perm][:, perm]
            result, n = lancoz_pruner(A_perm, target_eigenvalue)
            stop_iter_list.append(n)

    mean_iter = np.mean(stop_iter_list)
    print(f"When V={V}, p={p}, the mean number of iterations is:", mean_iter)
    return mean_iter


def experiment_1():
    """
    This function is the experiment 1 of the Lancoz algorithm.
    It generate a random matrix A and find the mean number of iterations before losing orthogonality.
    """
    np.random.seed(2024)
    mean_iter_list = []
    Vs = 200 * np.arange(1, 6)
    for V in Vs:
        p = 6 / V
        m = mean_iter_finder(V, p)
        mean_iter_list.append(m)
    with open("mean_iter_list.txt", "w") as f:
        f.write(str(Vs) + "\n")
        f.write(str(mean_iter_list))
    plt.figure(figsize=(10, 6))
    plt.plot(Vs, mean_iter_list, marker="o", label="mean_iter")
    plt.xlabel("Number of vertices")
    plt.ylabel("Mean of iterations before losing orthogonality")
    plt.title("Mean of iterations against the number of vertices")
    plt.legend()
    plt.savefig("./image/mean_iter_vs_V.png")
    plt.show()


def experiment_2():
    np.random.seed(2024)
    Vs = 200 * np.arange(1, 6)
    rs = [0.01, 0.02, 0.05, 0.1, 0.2]
    plt.figure(figsize=(10, 6))
    for r in rs:
        mean_stop_iter_list = []
        for V in Vs:
            p = 6 / V
            q = r * p  # perturbation rate
            stop_iter_list = []
            for _ in range(10):
                A = random_matrix_generator(
                    n=V,
                    nonzeros_prob=p,
                    weighted=True,
                    normalize=False,
                    symmetric=True,
                    fill_diagonal=False,
                    show_matrix=False,
                    save_fig=False,
                )
                target_eigenvalue = np.linalg.eigvalsh(A)
                for _ in range(10):
                    A_perturb = symmetric_matrix_perturbator(
                        A, permutation=True, perturbating_prob=q
                    )
                    result, n = lancoz_pruner(A_perturb, target_eigenvalue)
                    stop_iter_list.append(n)
            mean_iter = np.mean(stop_iter_list)
            print(
                f"When r={r}, V={V}, the mean number of stopping iterations is:",
                mean_iter,
            )
            mean_stop_iter_list.append(mean_iter)
        plt.plot(Vs, mean_stop_iter_list, marker="o", label=f"r={r*100:.0f}%")
    plt.xlabel("Number of vertices")
    plt.ylabel("Mean of stopping iterations")
    plt.title("Mean of stopping iterations against vertex number")
    plt.legend()
    plt.savefig("./image/stopping_iterations.png")
    plt.show()


def experiment_3():
    pass


if __name__ == "__main__":
    experiment_1()
    # experiment_2()
