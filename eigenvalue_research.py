"""
This is the code for the eigenvalue research.
We generate diiferrent types of random matrices and visualize their eigenvalues
to see which kind of adjacency matrix is suitable for pruning.
"""

from utils import *


def main():
    for weighted in [True, False]:
        for normalize in [True, False]:
            for symmetric in [True, False]:
                for fill_diagonal in [True, False]:
                    info = f"weighted-{weighted}, normalize-{normalize}, symmetric-{symmetric}, fill_diagonal-{fill_diagonal}"
                    print(info)
                    A = random_matrix_generator(
                        n=50,
                        nonzeros_prob=0.1,
                        weighted=weighted,
                        normalize=normalize,
                        symmetric=symmetric,
                        fill_diagonal=fill_diagonal,
                        show_matrix=True,
                        save_matrix=True,
                        save_dir="./image",
                    )
                    eigenvalue_visualizor(
                        A, title=info, save_fig=True, save_dir="./image"
                    )


if __name__ == "__main__":
    main()
