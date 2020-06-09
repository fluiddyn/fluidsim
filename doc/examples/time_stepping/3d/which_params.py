import numpy as np

coefficients = np.array([2 / 3, 0.8, 0.94])

nx_bigs = [800, 560, 420]


def compute(nx, nb_processes=20, verbose=True):

    nb_points = np.empty_like(coefficients, dtype=int)

    nb_points[0] = nx

    tmp = nx * coefficients[0] / coefficients[1:] / (2 * nb_processes) + 1e-10
    # print(tmp)
    # print(np.round(tmp))

    nb_points[1:] = 2 * nb_processes * np.round(tmp)

    if verbose:
        print(f"\n{nb_processes = }")
        print("nx =", nb_points)
        kmax = coefficients * deltak * nb_points / 2
        print(f"{kmax[0] = :.2f}")
        print(np.round(100 * (kmax - kmax[0]) / kmax[0], decimals=2))

    return nb_points


if __name__ == "__main__":
    compute(800)
    compute(560)
    compute(420, 10)
