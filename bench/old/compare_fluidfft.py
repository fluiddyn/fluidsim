#!/usr/bin/env python
"""
python compare_fluidfft.py
mpirun -np 8 python compare_fluidfft.py

"""
from time import time

import pstats
import cProfile

from fluidsim.solvers.ns2d.solver import Simul as Simul
from fluidsim.solvers.ns2d.solver_oper_cython import Simul as SimulOperCython


def modif_params(params, old=False):

    params.short_name_type_run = "profile"

    nh = 512 * 2
    params.oper.nx = nh
    params.oper.ny = nh
    lh = 6.0
    params.oper.Lx = lh
    params.oper.Ly = lh

    params.oper.coef_dealiasing = 2.0 / 3

    params.FORCING = True
    params.forcing.type = "tcrandom"
    params.forcing.nkmax_forcing = 5
    params.forcing.nkmin_forcing = 4
    params.forcing.forcing_rate = 1.0

    delta_x = lh / nh
    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8
    )

    try:
        params.f = 1.0
        params.c2 = 200.0
    except (KeyError, AttributeError):
        pass

    params.time_stepping.deltat0 = 1.0e-4
    params.time_stepping.USE_CFL = False

    params.time_stepping.it_end = 10
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0

    params.output.HAS_TO_SAVE = False
    # params.output.periods_save.phys_fields = 0.1
    # params.output.periods_save.spatial_means = 0.1
    # params.output.periods_save.spectra = 0.1
    # params.output.periods_save.spect_energy_budg = 0.
    # params.output.periods_save.increments = 0.

    if not old:
        params.short_name_type_run = "profile2"
        # params.oper.type_fft = 'fft2d.mpi_with_fftw1d'
    else:
        params.oper.type_fft = "FFTWCY"


params = SimulOperCython.create_default_params()
modif_params(params, "old")

sim_oper_cython = SimulOperCython(params)

t_start0 = time()
cProfile.runctx(
    "sim_oper_cython.time_stepping.start()",
    globals(),
    locals(),
    "profile_oper_cython.pstats",
)
t_end0 = time()


params = Simul.create_default_params()
modif_params(params)
sim = Simul(params)

t_start1 = time()
cProfile.runctx(
    "sim.time_stepping.start()", globals(), locals(), "profile.pstats"
)
t_end1 = time()


if sim.oper.rank == 0:
    s = pstats.Stats("profile_oper_cython.pstats")
    s.strip_dirs().sort_stats("time").print_stats(10)

    s = pstats.Stats("profile.pstats")
    s.strip_dirs().sort_stats("time").print_stats(10)

    print(
        "elapsed times: {:.3f} and {:.3f}".format(
            t_end0 - t_start0, t_end1 - t_start1
        )
    )

    print(
        "with gprof2dot and graphviz (command dot):\n"
        "gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png\n"
        "gprof2dot -f pstats profile_oper_cython.pstats | "
        "dot -Tpng -o profile_oper_cython.png"
    )
