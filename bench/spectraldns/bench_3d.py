"""Tiny benchmark of spectralDNS biperiodic 3D solver
=====================================================

A modified version of ``spectralDNS/demo/TG.py``

To run::

  python bench_3d.py --optimization cython NS

  mpirun -np 2 python bench_3d.py --optimization cython NS

To be compared with::

  fluidsim-bench 128 -d 3 -s ns3d -it 10

  mpirun -np 2 fluidsim-bench 128 -d 3 -s ns3d -it 10

"""
import time
import numpy as np
from numpy import zeros, sum, float64, sin, cos, prod, asscalar

# from numpy.linalg import norm
from spectralDNS import config, get_solver, solve

# import matplotlib.pyplot as plt
plt = None


def initialize(solver, context):
    if "NS" in config.params.solver:
        initialize1(solver, context)
    else:
        initialize2(solver, context)
    config.params.t = 0.0
    config.params.tstep = 0


def initialize1(solver, context):
    """Navier-Stokes solver initialization."""
    U, X = context.U, context.X
    U[0] = np.ones_like(X[0])
    U[1] = np.ones_like(X[0])
    U[2] = np.ones_like(X[0])
    U_hat = solver.set_velocity(**context)


def initialize2(solver, context):
    """Boussinesq solver initialization."""
    U, X = context.U, context.X
    U[0] = sin(X[0]) * cos(X[1]) * cos(X[2])
    U[1] = -cos(X[0]) * sin(X[1]) * cos(X[2])
    U[2] = 0
    U_hat = context.work[(context.W_hat, 0)]
    for i in range(3):
        U_hat[i] = context.FFT.fftn(U[i], U_hat[i])
    W_hat = solver.cross2(context.W_hat, context.K, U_hat)


def energy_fourier(comm, a):
    N = config.params.N
    result = (
        2 * sum(abs(a[..., 1:-1]) ** 2)
        + sum(abs(a[..., 0]) ** 2)
        + sum(abs(a[..., -1]) ** 2)
    )
    result = comm.allreduce(result)
    return result


k = []
w = []
im1 = None
kold = zeros(1)


def update(context):
    global k, w, im1
    c = context
    params = config.params
    solver = config.solver

    if (
        params.tstep % params.compute_energy == 0
        or params.tstep % params.plot_step == 0
        and params.plot_step > 0
    ):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if params.solver == "NS":
            P = solver.get_pressure(**c)

    if plt is not None:
        if (
            params.tstep % params.plot_step == 0
            and solver.rank == 0
            and params.plot_step > 0
        ):
            if im1 is None:
                plt.figure()
                im1 = plt.contourf(
                    c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100
                )
                plt.colorbar(im1)
                plt.draw()
                globals().update(im1=im1)
            else:
                im1.ax.clear()
                im1.ax.contourf(
                    c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100
                )
                im1.autoscale()
            plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        dx, L = params.dx, params.L
        ww = solver.comm.reduce(
            sum(curl.astype(float64) * curl.astype(float64)) / prod(params.N) / 2
        )
        # Compute energy with double precision
        kk = solver.comm.reduce(
            sum(U.astype(float64) * U.astype(float64)) / prod(params.N) / 2
        )
        if "shenfun" in params.solver:
            ww2 = energy_fourier(solver.comm, c.U_hat) / 2
        else:
            ww2 = energy_fourier(solver.comm, c.U_hat) / prod(params.N) ** 2 / 2

        kold[0] = kk
        if solver.rank == 0:
            k.append(kk)
            w.append(ww)
            print(params.t, float(kk), float(ww), float(ww2))


def regression_test(context):
    params = config.params
    solver = config.solver
    dx, L = params.dx, params.L
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    w = solver.comm.reduce(
        sum(curl.astype(float64) * curl.astype(float64)) / prod(params.N) / 2
    )
    # Compute energy with double precision
    k = solver.comm.reduce(
        sum(U.astype(float64) * U.astype(float64)) / prod(params.N) / 2
    )
    config.solver.MemoryUsage("End", solver.comm)
    if solver.rank == 0:
        assert round(asscalar(w) - 0.375249930801, params.ntol) == 0
        assert round(asscalar(k) - 0.124953117517, params.ntol) == 0


if __name__ == "__main__":
    Re = 1e4
    U = 2 ** (1.0 / 3)
    L = 1.0
    dt = 1e-12
    config.update(
        {
            "nu": U * L / Re,  # Viscosity
            "dt": dt,  # Time step
            "T": 11 * dt,  # End time
            "L": [L, L, L],
            "M": [7, 7, 7],  # Mesh size is pow(2, M[i]) in direction i
            #'planner_effort': {'fft': 'FFTW_EXHAUSTIVE'},
            #'decomposition': 'pencil',
            #'P1': 2
        },
        "triplyperiodic",
    )
    config.triplyperiodic.add_argument("--compute_energy", type=int, default=2)
    config.triplyperiodic.add_argument("--plot_step", type=int, default=2)
    if plt is None:
        sol = get_solver(mesh="triplyperiodic")
    else:
        sol = get_solver(
            update=update, regression_test=regression_test, mesh="triplyperiodic"
        )

    context = sol.get_context()

    # Add curl to the stored results. For this we need to update the update_components
    # method used by the HDF5Writer class to compute the real fields that are stored
    WRITE_HDF5 = False
    if config.params.solver == "NS" and WRITE_HDF5:
        context.hdf5file.fname = "NS8.h5"
        context.hdf5file.components["curlx"] = context.curl[0]
        context.hdf5file.components["curly"] = context.curl[1]
        context.hdf5file.components["curlz"] = context.curl[2]

        def update_components(**context):
            """Overload default because we want to store the curl as well"""
            U = sol.get_velocity(**context)
            P = sol.get_pressure(**context)
            curl = sol.get_curl(**context)

        context.hdf5file.update_components = update_components

    initialize(sol, context)

    # Double check benchmark walltime
    start_time = time.time()
    solve(sol, context)
    end_time = time.time()
    print("Run time: %f" % (end_time - start_time))

    # context.hdf5file._init_h5file(config.params, **context)
    # context.hdf5file.f.close()
