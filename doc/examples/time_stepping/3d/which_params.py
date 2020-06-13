import numpy as np

coefficients = np.array([2 / 3, 0.90, 1.0])

nx_bigs = [800, 560, 420]


def compute(nx, nb_processes=20, verbose=True):

    nb_points = np.empty_like(coefficients, dtype=int)

    nb_points[0] = nx

    tmp = nx * coefficients[0] / coefficients / (2 * nb_processes) + 1e-10

    nb_points[1:] = 2 * nb_processes * np.round(tmp[1:])

    if verbose:
        print(f"\n{nb_processes = }")
        print("coef dealisasing", coefficients)
        # print(tmp)
        # print(np.round(tmp))
        print("nx =", nb_points)
        kmax = coefficients * deltak * nb_points / 2
        print(f"{kmax[0] = :.2f}")
        print(np.round(100 * (kmax - kmax[0]) / kmax[0], decimals=2), "%")

    return nb_points


if __name__ == "__main__":
    compute(nx_bigs[0])
    compute(nx_bigs[1])
    compute(nx_bigs[2], 10)
