"""Run profiles (:mod:`fluidsim.util.console.profile`)
======================================================

"""

import gc
import os
from time import time
from pathlib import Path

import pstats
import cProfile

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluiddyn.io import stdout_redirected

from fluidsim import _is_testing

from ..util import import_module_solver_from_key
from .util import (
    modif_params2d,
    modif_params3d,
    init_parser_base,
    parse_args_dim,
    tear_down,
    get_path_file,
)

from .bench import get_opfft


path_results = "/tmp/fluidsim_profile"
old_print = print
print = mpi.printby0
rank = mpi.rank
nb_proc = mpi.nb_proc
description = "Profile time-elapsed in various function calls"


def run_profile(sim, nb_dim=None, path_results=None, plot=False, verbose=True):
    """Profile a simulation run and save the results in `profile.pstats`

    Parameters
    ----------
    sim : Simul
        An initialized simulation object
    nb_dim : int
        Dimension of the solver
    path_results : str
        Path where all pstats files will be saved

    """
    if nb_dim is None:
        try:
            nb_dim = len(sim.oper.axes)
        except AttributeError:
            raise ValueError

    if path_results is None:
        path_results = Path.cwd()
    else:
        path_results = Path(path_results)

    if path_results.name.endswith(".pstats"):
        path = path_results
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path, t_as_str = get_path_file(sim, path_results, "profile", ".pstats")
        path = Path(path)

    if mpi.rank > 0:
        path = path.with_name(path.stem + f"_rank{mpi.rank}" + path.suffix)

    with stdout_redirected(not verbose):
        sim.time_stepping.init_from_params()
        t0 = time()
        cProfile.runctx(
            "sim.time_stepping.main_loop()", globals(), locals(), path
        )
        t_end = time()
        sim.time_stepping.finalize_main_loop()

    if mpi.rank == 0:
        times = analyze_stats(path, nb_dim, plot)
        print(f"\nelapsed time = {t_end - t0:.3f} s")
        name_solver = sim.__module__.rsplit(".", maxsplit=1)[0]
        print(
            f"To display the results:\nfluidsim-profile -p -sf {path} -s {name_solver}"
        )
        print(
            "\nwith gprof2dot and graphviz (command dot):\n"
            f"gprof2dot -f pstats {path} | dot -Tpng -o profile.png"
        )

    else:
        # Retain only rank 0 profiles
        os.remove(path)

    return path


def profile(
    solver,
    dim="2d",
    n0=1024 * 2,
    n1=None,
    n2=None,
    path_dir=None,
    type_fft=None,
    verbose=False,
    plot=False,
    it_end=None,
):
    """Instantiate simulation object and run profiles."""

    inputs = dict(solver=solver, n0=n0, n1=n1, type_fft=type_fft)

    if dim == "3d":
        inputs["n2"] = n2

    print("profile with inputs:")
    for k, v in inputs.items():
        print(k, "=", str(v))

    def _profile(type_fft):
        """Run profile once."""
        Simul = solver.Simul
        params = Simul.create_default_params()

        if dim == "2d":
            modif_params2d(
                params, n0, n1, name_run="bench", type_fft=type_fft, it_end=it_end
            )
        elif dim == "3d":
            modif_params3d(
                params,
                n0,
                n1,
                n2,
                name_run="bench",
                type_fft=type_fft,
                it_end=it_end,
            )
        else:
            raise ValueError("dim has to be in ['2d', '3d']")

        nb_dim = int(dim[0])

        with stdout_redirected(not verbose):
            sim = Simul(params)

        try:
            run_profile(sim, nb_dim, path_dir, verbose=verbose)
        except Exception as e:
            if _is_testing:
                raise

            else:
                print(
                    "WARNING: Some error occurred while running benchmark"
                    " / saving results!"
                )
                raise

                print(e)
        finally:
            tear_down(sim)
            gc.collect()

    if str(type_fft).lower() == "all":
        d = get_opfft(n0, n1, n2, dim, only_dict=True)
        for type_fft, cls in d.items():
            if cls is not None:
                print(type_fft)
                _profile(type_fft)
    else:
        _profile(type_fft)


def init_parser(parser):
    """Initialize argument parser for `fluidsim profile`."""

    init_parser_base(parser)
    parser.add_argument("-o", "--output_dir", default=path_results)
    parser.add_argument(
        "-t",
        "--type-fft",
        default=None,
        help=(
            'specify FFT type key (for eg. "fft2d.mpi_with_fftw1d") or "all";'
            "if not specified uses the default FFT method in operators"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-sf", "--stats_file", default=None)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument(
        "-it", "--it-end", default=10, type=int, help="Number of iterations"
    )


def run(args):
    """Run `fluidsim profile` command."""

    if args.stats_file is not None and args.solver is None:
        # get solver from name... Not clean at all...
        tmp = os.path.split(args.stats_file)[-1]
        args.solver = tmp.split("result_profile_")[-1].split("_")[0]

    args = parse_args_dim(args)

    if args.stats_file is not None:
        nb_dim = int(args.dim.split("d")[0])
        analyze_stats(args.stats_file, nb_dim, args.plot)
        return

    # Initialize simulation and run benchmarks
    solver = import_module_solver_from_key(args.solver)
    profile(
        solver,
        args.dim,
        args.n0,
        args.n1,
        args.n2,
        path_dir=args.output_dir,
        type_fft=args.type_fft,
        verbose=args.verbose,
        plot=args.plot,
        it_end=args.it_end,
    )


def _compute_shorter_name(key, kind):
    if kind == "fft_as":
        words = key.split()
        if len(words) == 1:
            return key

        else:
            return words[1][1:-1]

    if kind == "pythran" in key:
        if "_pythran." in key:
            key = key.split("_pythran.")[-1].rstrip(">") + " (pythran)"
        else:
            key = key.split("<built-in method ")[-1].rstrip(">")
        return key

    if "fluidsim.solvers." in key:
        return key.split("fluidsim.solvers.")[-1][:-1]

    if key.startswith("<built-in method "):
        return key.split("<built-in method ")[-1][:-1]

    if key.startswith("<method "):
        key = key.split("<method ")[-1][:-1]
        return key.split(" ")[0].strip("'")

    return key


def plot_pie(
    times,
    long_functions,
    ax=None,
    times_descending=False,
    for_latex=False,
    **kwargs,
):
    """Plot a pie chart"""
    percentages = []
    labels = []
    for k, v in long_functions.items():
        percentages.append(v["percentage"])
        name = _compute_shorter_name(k, v["kind"])
        if for_latex:
            name = r"\texttt{" + name.replace("_", r"\_") + r"}"
        labels.append(name)

    percentages.append(100 - sum(percentages))
    percentages = np.array(percentages)
    labels.append("other")

    labels = np.array(labels)
    if times_descending:
        args = percentages.argsort()[::-1]
        percentages.sort()
        percentages = percentages[::-1]
    else:
        args = percentages.argsort()
        percentages.sort()

    labels = labels[args]
    if "explode" in kwargs:
        if kwargs["explode"] == "other":
            # Explode the 'other' slice from the pie chart
            idx_other = np.argwhere(labels == "other")[0]
            explode = np.zeros_like(labels, dtype=float)
            explode[idx_other] = 0.2
            kwargs["explode"] = explode
        elif isinstance(kwargs["explode"], float):
            # Explode all slices based on percentage
            kwargs["explode"] /= percentages

    if "labeldistance" in kwargs:
        if isinstance(kwargs["labeldistance"], float):
            kwargs["labeldistance"] /= percentages
            kwargs["labeldistance"] += 1.1

    if ax is None:
        fig, ax = plt.subplots()

    if "startangle" not in kwargs:
        kwargs["startangle"] = 0

    for label, perc in zip(labels, percentages):
        print(
            "(label, perc) = ({:40s} {:5.2f} %)".format(repr(label) + ",", perc)
        )

    autopct = "%1.1f\\%%" if for_latex else "%1.1f%%"
    pie = ax.pie(percentages, labels=labels, autopct=autopct, **kwargs)
    ax.axis("equal")

    return pie


_kinds = ("fft_as", "pythran", ".pyx", ".py", "built-in", "numpy")


def analyze_stats(
    path, nb_dim=2, plot=False, threshold_long_function=0.01, verbose=True
):
    """Print analysis of profiling result of a 2D solver.

    Parameters
    ----------
    s : pstats.Stats
        Object pointing to a stats file

    """
    stats = pstats.Stats(str(path))
    if verbose:
        stats.sort_stats("time").print_stats(20)

    if nb_dim not in (2, 3):
        raise NotImplementedError

    total_time = 0.0
    for key, value in stats.stats.items():
        time = value[2]
        total_time += time

    long_functions = {}

    key_fft = f"fft{nb_dim}d"
    kinds = _kinds + (key_fft,)
    times = {k: 0.0 for k in kinds}

    for key, value in stats.stats.items():
        name = key[2]
        time = value[2]

        for k in kinds:
            if k in name or k in key[0]:
                if k == ".pyx":
                    if "fft/Sources" in key[0]:
                        continue

                    if "fft_as_arg" in name:
                        continue

                if k == key_fft:
                    if (
                        "__pythran__" in name
                        or "operators.py" in key[0]
                        or "fft_as_arg" in name
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

                if k == "fft_as" and ".pyx" in key[0]:
                    continue

                if k == ".py" and "fft_as_arg" in name:
                    continue

                if k == "built-in" and "pythran" in name:
                    continue

                times[k] += time

                if time / total_time > threshold_long_function:
                    long_functions[name] = dict(
                        time=time, percentage=100 * time / total_time, kind=k
                    )

    if plot:
        plot_pie(times, long_functions)
        plt.show()

    if not verbose:
        return times, long_functions

    print(
        "\nlong_functions (more than {} % of total time):".format(
            100 * threshold_long_function
        )
    )
    for name, d in long_functions.items():
        print(name + ":")
        for k, v in d.items():
            if k == "percentage":
                print("    " + k + f" = {v:.2f} %")
            else:
                print("    " + k + " = " + repr(v))

    print("\nAnalysis (percentage of total time):")

    keys = list(times.keys())
    keys.sort(key=lambda key: times[key], reverse=True)

    for k in keys:
        t = times[k]
        if t > 0:
            print(
                "time {:10s}: {:7.03f} % ({:4.02f} s)".format(
                    k, t / total_time * 100, t
                )
            )

    print(
        "-" * 26
        + "\n{:15s}  {:7.03f} %".format(
            "", sum([t for t in times.values()]) / total_time * 100
        )
    )

    del times[".py"]
    time_in_not_py = sum([t for t in times.values()])
    print(
        "In not Python functions:\n{:15s}".format("")
        + "  {:7.03f} %".format(time_in_not_py / total_time * 100)
    )

    return times, long_functions
