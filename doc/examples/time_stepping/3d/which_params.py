import numpy as np

try:
    from sympy import primefactors
except ImportError:
    print("ImportError: sympy")
    pass

L = 1.0
V0 = 1.0
Re = 1600
Lx = 2 * np.pi * L
deltak = 2 * np.pi / Lx
nu = V0 * L / Re

# A result from a simulation
epsilon = 0.012
# epsilon = 0.008
eta = (nu**3 / epsilon) ** (1 / 4)

coefficients = np.array([2 / 3, 0.90, 1.0])

nx_bigs = [800, 640, 400, 256, 128]
nb_procs = [20, 10, 10, 4, 2]


def compute(nx, nb_processes=20, verbose=True):

    nb_points = np.empty_like(coefficients, dtype=int)

    nb_points[0] = nx

    tmp = nx * coefficients[0] / coefficients / (2 * nb_processes) + 1e-10
    round_tmp = np.round(tmp)
    if 13 in round_tmp:
        index = np.where(round_tmp == 13)[0][0]
        round_tmp[index] = 14

    # if 11 in round_tmp:
    #     index = np.where(round_tmp == 11)[0][0]
    #     round_tmp[index] = 10

    nb_points[1:] = 2 * nb_processes * round_tmp[1:]

    if verbose:
        factors = []
        for number in round_tmp:
            factors.append(primefactors(int(number)))
        print(f"\n{nb_processes = }")
        print("nx =", nb_points)
        kmax = coefficients * deltak * nb_points / 2
        print(f"{kmax[0]*eta = :.2f}")
        print(tmp)
        print(round_tmp)
        print(f"{factors = }")
        print(np.round(100 * (kmax - kmax[0]) / kmax[0], decimals=2), "%")

    return nb_points


if __name__ == "__main__":

    print("coef dealisasing", coefficients)

    for nx, nb_proc in zip(nx_bigs, nb_procs):
        compute(nx, nb_proc)
