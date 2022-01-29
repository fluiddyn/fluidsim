"""Tiny benchmark of spectralDNS biperiodic 2D solver
=====================================================

A modified version of ``spectralDNS/demo/TG2D.py``

To run::

  python bench_2d.py --optimization cython NS2D

  mpirun -np 2 python bench_2d.py --optimization cython NS2D


To be compared with::

  fluidsim-bench 512 -d 2 -s ns2d -it 10
  mpirun -np 2 fluidsim-bench 512 -d 2 -s ns2d -it 10

  fluidsim-bench 1024 -d 2 -s ns2d -it 10
  mpirun -np 2 fluidsim-bench 1152 -d 2 -s ns2d -it 10

"""
import time
import numpy as np
from spectralDNS import config, get_solver, solve

# import matplotlib.pyplot as plt
plt = None  # Disables plotting


def initialize(U, U_hat, X, FFT, **context):
    U[0] = np.ones_like(X[0])
    U[1] = np.ones_like(X[0])
    for i in range(2):
        U_hat[i] = FFT.fft2(U[i], U_hat[i])
    config.params.t = 0.0
    config.params.tstep = 0


im = None


def update(context):
    global im
    params = config.params
    solver = config.solver

    # initialize plot
    if params.tstep == 1:
        im = plt.imshow(np.zeros((params.N[0], params.N[1])))
        plt.colorbar(im)
        plt.draw()

    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        curl = solver.get_curl(**context)
        im.set_data(curl[:, :])
        im.autoscale()
        plt.pause(1e-6)


if __name__ == "__main__":
    Re = 1e4
    U = 2**0.5
    L = 1.0
    config.update(
        {
            "nu": U * L / Re,
            "dt": 1e-12,
            "T": 1e-11,  # Should run 10 iterations
            "write_result": 100,
            "L": [L, L],
            "M": [10, 10]  # Mesh size is pow(2, M[i]) in direction i
            # 2**9 == 512
        },
        "doublyperiodic",
    )

    # required to allow overloading through commandline
    config.doublyperiodic.add_argument("--plot_result", type=int, default=10)
    if plt is None:
        sol = get_solver(mesh="doublyperiodic")
    else:
        sol = get_solver(update=update, mesh="doublyperiodic")

    context = sol.get_context()
    initialize(**context)

    # Double check benchmark walltime
    start_time = time.time()
    solve(sol, context)
    end_time = time.time()
    print("Run time: %f" % (end_time - start_time))
