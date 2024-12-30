"""
This is the code for the eigenvalue research.
We generate diiferrent types of random matrices and visualize their eigenvalues
to see which kind of adjacency matrix is suitable for pruning.
"""

from utils import *


def main():
    np.random.seed(2024)
    for weighted in [True, False]:
        for normalize in [True, False]:
            for symmetric in [True, False]:
                for fill_diagonal in [True, False]:
                    if normalize and symmetric:
                        print("Normalize and symmetric cannot be both True.")
                    info = f"weighted-{weighted}, normalize-{normalize}, symmetric-{symmetric}, fill_diagonal-{fill_diagonal}"
                    print(info)
                    A = random_matrix_generator(
                        n=100,
                        nonzeros_prob=6/100,
                        weighted=weighted,
                        normalize=normalize,
                        symmetric=symmetric,
                        fill_diagonal=fill_diagonal,
                        show_matrix=True,
                        save_fig=True,
                        save_dir="./image",
                    )
                    eigenvalue_visualizor(
                        A, title=info, save_fig=True, save_dir="./image"
                    )


if __name__ == "__main__":
    main()
