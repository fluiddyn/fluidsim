import os
from time import time
import pstats
import cProfile
import json
import socket

from fluiddyn.util import time_as_str
from fluiddyn.util import mpi


def modif_params_profile2d(params, nh=3**2 * 2**7):
    params.short_name_type_run = "profile"

    params.oper.nx = nh
    params.oper.ny = nh

    # params.oper.type_fft = 'fft2d.mpi_with_fftw1d'
    # params.oper.type_fft = 'fft2d.with_cufft'

    params.forcing.enable = True
    params.forcing.type = "tcrandom"
    params.forcing.nkmax_forcing = 5
    params.forcing.nkmin_forcing = 4
    params.forcing.forcing_rate = 1.0

    params.nu_8 = 1.0
    try:
        params.f = 1.0
        params.c2 = 200.0
    except AttributeError:
        pass

    try:
        params.N = 1.0
    except AttributeError:
        pass

    params.time_stepping.deltat0 = 1.0e-4
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = 10
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0
    params.output.HAS_TO_SAVE = 0


def modif_params_profile3d(params, nh=256, nz=32):
    params.short_name_type_run = "profile"

    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.nz = nz

    # params.oper.type_fft = 'fft3d.with_fftw3d'
    params.oper.type_fft = "fft3d.with_cufft"

    # params.forcing.enable = False
    # params.forcing.type = 'tcrandom'
    # params.forcing.nkmax_forcing = 5
    # params.forcing.nkmin_forcing = 4
    # params.forcing.forcing_rate = 1.

    params.nu_8 = 1.0
    try:
        params.f = 1.0
        params.c2 = 200.0
    except AttributeError:
        pass

    try:
        params.N = 1.0
    except AttributeError:
        pass

    params.time_stepping.deltat0 = 1.0e-4
    params.time_stepping.USE_CFL = False
    params.time_stepping.it_end = 10
    params.time_stepping.USE_T_END = False

    params.output.periods_print.print_stdout = 0
    params.output.HAS_TO_SAVE = 0


def profile(sim, nb_dim=2):

    t0 = time()

    cProfile.runctx(
        "sim.time_stepping.start()", globals(), locals(), "profile.pstats"
    )
    t_end = time()
    if sim.oper.rank == 0:
        s = pstats.Stats("profile.pstats")
        # s.strip_dirs().sort_stats('time').print_stats(16)
        s.sort_stats("time").print_stats(12)

        if nb_dim == 2:
            times = print_analysis(s)
        elif nb_dim == 3:
            times = print_analysis3d(s)
        else:
            raise NotImplementedError

        print("\nelapsed time = {:.3f} s".format(t_end - t0))

        print(
            "\nwith gprof2dot and graphviz (command dot):\n"
            "gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png"
        )


def bench(sim):

    t_as_str = time_as_str()
    t0 = time()
    sim.time_stepping.start()
    t_elapsed = time() - t0

    if sim.oper.rank != 0:
        return

    path_results = "results_bench"
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    key_solver = sim.info_solver.short_name.lower()
    pid = os.getpid()
    nfile = (
        "result_bench_"
        + key_solver
        + "_"
        + sim.oper.produce_str_describing_grid()
        + "_"
        + t_as_str
        + f"_{pid}"
        + ".json"
    )

    path = os.path.join(path_results, nfile)

    results = {
        "t_elapsed": t_elapsed,
        "key_solver": sim.info_solver.short_name.lower(),
        "nx": sim.oper.nx,
        "ny": sim.oper.ny,
        "nb_proc": mpi.nb_proc,
        "pid": pid,
        "time_as_str": t_as_str,
        "hostname": socket.gethostname(),
    }

    try:
        results["nz"] = sim.oper.nz
    except AttributeError:
        pass

    with open(path, "w") as file:
        json.dump(results, file, sort_keys=True)
        file.write("\n")


def print_analysis(s):
    total_time = 0.0
    times = {"fft2d": 0.0, "fft_as": 0.0, "pythran": 0.0, ".pyx": 0.0}
    for key, value in s.stats.items():
        name = key[2]
        time = value[2]
        total_time += time
        if name == "one_time_step_computation":
            print(
                "warning: special case one_time_step_computation "
                "included in .pyx (see explanation in the code)"
            )
            times[".pyx"] += time

        for k in times.keys():
            if k in name or k in key[0]:
                if k == ".pyx":
                    if "fft/Sources" in key[0]:
                        continue
                    if "fft_as_arg" in key[2]:
                        continue

                if k == "fft2d":

                    if (
                        "util_pythran" in key[2]
                        or "operators.py" in key[0]
                        or "fft_as_arg" in key[2]
                    ):
                        continue

                    callers = value[4]

                    time = 0
                    for kcaller, vcaller in callers.items():
                        if (
                            "fft_as_arg" not in kcaller[2]
                            and "fft_as_arg" not in kcaller[0]
                        ):
                            time += vcaller[2]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                if k == "fft_as":
                    if ".pyx" in key[0]:
                        continue
                    # time = value[3]

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                times[k] += time

    print("Analysis (percentage of total time):")

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        print(
            "time {:10s}: {:5.01f} % ({:4.02f} s)".format(
                k, t / total_time * 100, t
            )
        )

    print(
        "-" * 24
        + "\n{:15s}  {:5.01f} %".format(
            "", sum([t for t in times.values()]) / total_time * 100
        )
    )

    return times


def print_analysis3d(s):
    total_time = 0.0
    times = {"fft3d": 0.0, "fft_as": 0.0, "pythran": 0.0, ".pyx": 0.0}
    for key, value in s.stats.items():
        name = key[2]
        time = value[2]
        total_time += time
        if name == "one_time_step_computation":
            print(
                "warning: special case one_time_step_computation "
                "included in .pyx (see explanation in the code)"
            )
            times[".pyx"] += time

        for k in times.keys():
            if k in name or k in key[0]:

                if k == "fft3d":
                    if "pythran" in key[0] or "pythran" in key[2]:
                        continue
                    if "operators.py" in key[0]:
                        continue
                    if "as_arg" in key[2]:
                        continue

                    # print(k, key)
                    # print(value[:100])
                    # print(time, '\n')

                times[k] += time

    print("Analysis (percentage of total time):")

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        print(
            "time {:10s}: {:5.01f} % ({:4.02f} s)".format(
                k, t / total_time * 100, t
            )
        )

    print(
        "-" * 24
        + "\n{:15s}  {:5.01f} %".format(
            "", sum([t for t in times.values()]) / total_time * 100
        )
    )

    return times
