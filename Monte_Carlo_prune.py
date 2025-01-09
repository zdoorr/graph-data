from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

IMAGES_DIR = "./image"


def experiment_1():
    """
    研究三种不同的距离度量函数对收敛的表示能力
    """
    # 先来一个示意图
    n = 200
    p = 6 / n
    G1, G2 = generate_graphs(n, p_edge=p, is_isomorphic=True, is_directed=False)
    total_samples = 20 * n
    initial_samples_count = samples_count_initializer(G1, total_samples)
    max_steps = 50  # 随机游走的最大步数
    threshold = 0
    dist_func_list = [normalized_2_norm, js_divergence, wasserstein_distance]
    for dist_func in dist_func_list:
        dist_list, final_samples_count, final_step = (
            Monte_Carlo_particle_transition_simulation(
                G1,
                initial_samples_count,
                max_steps,
                threshold,
                dist_func=dist_func,
                trans_to_itself=False,
                SEED=2025,
            )
        )
        dist_list = np.array(dist_list)
        # 除以最大的
        dist_list = dist_list / np.max(dist_list)
        plt.plot(np.array(dist_list), label=dist_func.__name__)
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("Normalized dist(x_n, x_n+1)")
    plt.title("Distance metric function's effect on convergence")
    # plt.savefig(os.path.join(IMAGES_DIR, "convergence_effect.png"))
    plt.show()


def mean_dist_update_when_converging(n, p, total_samples, dist_func=js_divergence):
    """
    给定图的大小，图生成边的概率，初始样本总数，距离度量函数，
    返回稳定时距离度量函数的均值
    """

    max_steps = 50  # 随机游走的最大步数
    threshold = 0

    mean_dist_list = []
    print("通过20轮循环估计稳态向量更新的均值")
    for _ in range(20):
        G1, G2 = generate_graphs(n, p_edge=p, is_isomorphic=True, is_directed=False)
        initial_samples_count = samples_count_initializer(G1, total_samples)

        dist_list, final_samples_count, final_step = (
            Monte_Carlo_particle_transition_simulation(
                G1,
                initial_samples_count,
                max_steps,
                threshold,
                dist_func=dist_func,
                trans_to_itself=False,
            )
        )
        dist_list_trunc = np.array(dist_list)[-5:]
        mean_dist_list.append(np.mean(dist_list_trunc))

    print(mean_dist_list)

    return np.mean(mean_dist_list)


def experiment_2():
    """
    研究迭代停止阈值随着n的变化
    """
    threshold_list = []
    n_list = list(range(25, 325, 25))
    for n in n_list:

        p_edge = 6 / n
        total_samples = 20 * n
        dist_func = js_divergence

        convergence_threshold = mean_dist_update_when_converging(
            n, p_edge, total_samples, dist_func
        )
        print(f"n={n}, convergence_threshold={convergence_threshold}")
        threshold_list.append(convergence_threshold)

    plt.plot(
        n_list,
        threshold_list,
        label="convergence threshold",
    )
    plt.xlabel("n")
    plt.ylabel("Convergence threshold")
    plt.title("Convergence threshold when n changes")
    plt.legend()
    plt.savefig(os.path.join(IMAGES_DIR, "n_effect_on_threshold.png"))
    plt.show()


def experiment_3():
    """
    稳态向量距离随着扰动比例的变化
    """
    n = 200
    p_edge = 6 / n
    total_samples = 20 * n
    dist_func = js_divergence
    convergence_threshold = mean_dist_update_when_converging(
        n, p_edge, total_samples, dist_func
    )
    print("The estimated convergence threshold is:", convergence_threshold)

    diff_matrix = []
    k_list = list(range(2, 22, 2))
    for k in k_list:
        diff_list = []
        for _ in range(50):
            G1, G2 = generate_graphs(
                n, p_edge=p_edge, is_isomorphic=False, is_directed=False, k=k
            )
            initial_samples_count_1 = samples_count_initializer(G1, total_samples)
            initial_samples_count_2 = samples_count_initializer(G2, total_samples)

            _, final_samples_count_1, _ = Monte_Carlo_particle_transition_simulation(
                G=G1,
                initial_samples_count=initial_samples_count_1,
                threshold=convergence_threshold,
                dist_func=dist_func,
            )
            _, final_samples_count_2, _ = Monte_Carlo_particle_transition_simulation(
                G=G2,
                initial_samples_count=initial_samples_count_2,
                threshold=convergence_threshold,
                dist_func=dist_func,
            )
            p = np.array(list(final_samples_count_1.values()))
            p = np.sort(p) / np.sum(p)

            q = np.array(list(final_samples_count_2.values()))
            q = np.sort(q) / np.sum(q)

            dist = dist_func(p, q)
            print(dist)
            diff_list.append(dist)
        print(f"k={k}, mean diff={np.mean(diff_list)}")
        print(f"k={k}, std diff={np.std(diff_list)}")
        diff_matrix.append(np.array(diff_list))

    # transform diff_matrix to dataframe
    diff_matrix = np.array(diff_matrix).T
    diff_df = pd.DataFrame(diff_matrix, columns=k_list)

    # 用seaborn根据diff_matrix画出violin图
    plt.figure(figsize=(16, 6))
    sns.violinplot(data=diff_df, palette="Set2")
    plt.xlabel("k")
    plt.ylabel("Distance between converged distributions")
    plt.title("Distance between converged distributions when k changes")
    plt.savefig(os.path.join(IMAGES_DIR, "k_effect_on_diff.png"))
    plt.show()


def experiment_4():
    """
    稳态向量距离随着扰动比例的变化，但让扰动比例在更大的范围内波动
    """
    n = 200
    p_edge = 6 / n
    total_samples = 20 * n
    dist_func = js_divergence
    convergence_threshold = mean_dist_update_when_converging(
        n, p_edge, total_samples, dist_func
    )
    print("The estimated convergence threshold is:", convergence_threshold)

    diff_matrix = []
    k_list = [20, 40, 60, 120, 240, 360]
    for k in k_list:
        diff_list = []
        for _ in range(50):
            G1, G2 = generate_graphs(
                n, p_edge=p_edge, is_isomorphic=False, is_directed=False, k=k
            )
            initial_samples_count_1 = samples_count_initializer(G1, total_samples)
            initial_samples_count_2 = samples_count_initializer(G2, total_samples)

            _, final_samples_count_1, _ = Monte_Carlo_particle_transition_simulation(
                G=G1,
                initial_samples_count=initial_samples_count_1,
                threshold=convergence_threshold,
                dist_func=dist_func,
            )
            _, final_samples_count_2, _ = Monte_Carlo_particle_transition_simulation(
                G=G2,
                initial_samples_count=initial_samples_count_2,
                threshold=convergence_threshold,
                dist_func=dist_func,
            )
            p = np.array(list(final_samples_count_1.values()))
            p = np.sort(p) / np.sum(p)

            q = np.array(list(final_samples_count_2.values()))
            q = np.sort(q) / np.sum(q)

            dist = dist_func(p, q)
            print(dist)
            diff_list.append(dist)
        print(f"k={k}, mean diff={np.mean(diff_list)}")
        print(f"k={k}, std diff={np.std(diff_list)}")
        diff_matrix.append(np.array(diff_list))

    # transform diff_matrix to dataframe
    diff_matrix = np.array(diff_matrix).T
    diff_df = pd.DataFrame(diff_matrix, columns=k_list)

    # 用seaborn根据diff_matrix画出vioolin图
    plt.figure(figsize=(16, 6))
    sns.violinplot(data=diff_df, palette="Set2")
    plt.xlabel("k")
    plt.ylabel("Distance between converged distributions")
    plt.title("Distance between converged distributions when k changes")
    plt.savefig(os.path.join(IMAGES_DIR, "k_effect_on_diff_large_scale.png"))
    plt.show()


if __name__ == "__main__":
    # experiment_1()
    # experiment_2()
    # experiment_3()
    experiment_4()
