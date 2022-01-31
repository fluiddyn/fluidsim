"""Load and plot benchmarks (:mod:`fluidsim.util.console.bench_analysis`)
=========================================================================

"""
import os
from glob import glob
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .bench import path_results, parse_args_dim, init_parser_base, ConsoleError

description = "Plot results of benchmarks"


def load_bench(path_dir, solver, hostname="any"):
    """Load benchmarks results saved as JSON files."""

    if solver.startswith("fluidsim"):
        solver = solver.split(".", 1)[1]

    dicts = []
    for path in glob(path_dir + f"/result_bench_{solver}*.json"):
        with open(path) as file:
            d = json.load(file)

        if hostname != "any" and not d["hostname"].startswith(hostname):
            continue

        dicts.append(d)

    df = pd.DataFrame(dicts)
    df = df[df.columns.difference(["pid", "time_as_str"])]
    return df


def filter_by_shape(df, n0, n1, n2=None):
    """Filters all results with the same `n0` and same `n1`."""

    if n2 is None:
        df = df[(df.n0 == n0) & (df.n1 == n1)]
        return df[df.columns.difference(["n0", "n1"])]

    else:
        df = df[(df.n0 == n0) & (df.n1 == n1) & (df.n2 == n2)]
        return df[df.columns.difference(["n0", "n1", "n2"])]


def filter_by_shapeloc(df, n0_loc, n1_loc, n2_loc=None):
    """Filters all results with the same `n0_loc * n1_loc`.

    This implies shapeK_loc has same no. of points. This is a weak check and
    the shapeK_loc may not have the same shape. Make sure all shapes are
    estimated and tested apriori using:

    >>> fluidsim bench -e
    >>> mpirun -np {nb_proc} fluidsim bench -c

    """
    if n2_loc is None:
        df = df[(df.n0_loc * df.n1_loc == n0_loc * n1_loc)]
        return df[df.columns.difference(["n0_loc", "n1_loc"])]

    else:
        df = df[(df.n0_loc * df.n1_loc * df.n2_loc == n0_loc * n1_loc * n2_loc)]
        return df[df.columns.difference(["n0_loc", "n1_loc", "n2_loc"])]


def exit_if_empty(df, input_params):
    """Check if the dataframe is empty."""
    if df.empty:
        print("No benchmarks corresponding to the input parameters:")
        for k, v in input_params.items():
            print(k, "=", repr(v))
        sys.exit()


def plot_scaling(
    path_dir,
    solver,
    hostname,
    n0,
    n1,
    n2=None,
    show=True,
    type_time="usr",
    type_plot="strong",
    fig=None,
    ax0=None,
    ax1=None,
    name_dir=None,
    once=False,
    t_min_cum=1e10,
):
    """Plot speedup vs number of processes from benchmark results."""

    input_params = dict(
        path_dir=path_dir, solver=solver, hostname=hostname, n0=n0, n1=n1
    )

    if name_dir is None:
        name_dir = (
            os.path.basename(os.path.abspath(path_dir)).replace("_", " ").upper()
        )

    df = load_bench(path_dir, solver, hostname)
    exit_if_empty(df, input_params)
    df.t_elapsed_sys /= df.nb_iter
    df.t_elapsed_usr /= df.nb_iter

    if type_plot == "strong":
        df_filter = filter_by_shape(df, n0, n1, n2)
    elif type_plot == "weak":
        df_filter = filter_by_shapeloc(df, n0, n1, n2)
    else:
        raise ConsoleError("Unknown plot type.")

    def group_df(df):
        """Group and take median dataframe results with same number of processes."""
        # for "scaling" (mpi)
        df = df[df.nb_proc > 1]
        exit_if_empty(df, input_params)
        nb_proc_min = df.nb_proc.min()
        df_grouped = df.groupby(["type_fft", "nb_proc"]).quantile(q=0.2)
        if show:
            print(df)
        return df_grouped, nb_proc_min

    keys_filter = [
        k for k in df_filter.columns if k.startswith("t_elapsed_" + type_time)
    ]

    df_filter, nb_proc_min_filter = group_df(
        df_filter[keys_filter + ["type_fft", "nb_proc"]]
    )
    df_filter = df_filter[keys_filter]
    df_filter_nb_proc_min = df_filter.xs(nb_proc_min_filter, level=1)

    def get_min(df):
        """Get minumum time from the whole dataframe"""
        m = df.values
        i0, i1 = np.unravel_index(np.argmin(m), m.shape)
        mymin = m[i0, i1]
        ind = df.index[i0]
        key = df.columns[i1]
        return mymin, ind, key

    t_min_filter, name_min_filter, key_min_filter = get_min(df_filter_nb_proc_min)

    print(
        "{}: Best for 2 procs t={} for {}, {}".format(
            path_dir, t_min_filter, name_min_filter, key_min_filter
        )
    )

    t_min_filter = min(t_min_filter, t_min_cum)
    # Finally, start preparing figure and plot
    if fig is None or ax0 is None or ax1 is None:
        fig, axes = plt.subplots(1, 2)
        ax0, ax1 = axes.ravel()

    ax0.set_ylabel(f"speedup ({type_plot} scaling)")
    ax1.set_ylabel(f"efficiency % ({type_plot} scaling)")

    def plot_once(ax, x, y, label, linestyle="-k"):
        """Avoid plotting the same label again."""
        plotted = any([lines.get_label() == label for lines in ax.get_lines()])
        if not (plotted and once):
            ax.plot(x, y, linestyle, label=label)

    def add_hline(ax, series=None, y=None):
        """Add a horizontal line to the axis."""
        if y is None:
            y = series.iloc[0]

        xmin = series.index.min()
        xmax = max(series.index.max(), ax.get_xlim()[1])
        # plot_once(ax, [xmin, xmax], [y, y], 'linear')
        ax.plot([xmin, xmax], [y, y], "k-")

    def set_label_scale(ax, yscale="log"):
        """Set log scale, legend and label of the axis."""
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("number of processes")

    # Plot speedup
    print("Grouped names:")
    for name in df_filter.index.levels[0]:
        tmp = df_filter.loc[name]
        print(name)

        for k in keys_filter:
            speedup = t_min_filter / tmp[k] * nb_proc_min_filter
            ax0.plot(
                speedup.index,
                speedup.values,
                "x-",
                label="{}, {}".format(name.replace("fluidfft.", ""), name_dir),
            )

    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    if type_plot == "strong":
        theoretical = [speedup.index.min(), speedup.index.max()]
        # plot_once(ax0, theoretical, theoretical, 'linear')
        ax0.plot(theoretical, theoretical, "k-")
    else:
        add_hline(ax0, speedup, 2)

    set_label_scale(ax0)
    ax0.legend()

    # Plot efficiency
    for name in df_filter.index.levels[0]:
        tmp = df_filter.loc[name]
        for k in keys_filter:
            speedup = t_min_filter / tmp[k] * nb_proc_min_filter
            if type_plot == "strong":
                efficiency = speedup / speedup.index * 100
            else:
                # efficiency = speedup / speedup.iloc[0] * 100
                efficiency = speedup / 2 * 100
            ax1.plot(
                efficiency.index,
                efficiency.values,
                "x-",
                label=f"{name}, {name_dir}",
            )

    add_hline(ax1, efficiency, 100)

    set_label_scale(ax1, "linear")

    title_dim = f"dim {n0}x{n1}"
    if n2 is not None:
        title_dim += f"x{n2}"

    fig.suptitle(
        f"Best for {nb_proc_min_filter} processes and {title_dim} :\n"
        f"{name_min_filter}, {key_min_filter}={t_min_filter * 1000:.2f} ms"
    )

    if show:
        plt.show()
    return fig, t_min_filter


def init_parser(parser):
    """Initialize argument parser for `fluidsim bench-analysis`."""

    init_parser_base(parser)
    parser.add_argument(
        "-i",
        "--input-dir",
        default=path_results,
        help="plot results from a single directory",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[],
        help="plot results from mulitiple input directories",
    )
    parser.add_argument(
        "-p",
        "--type-plot",
        default="strong",
        help="load and plot data for strong or weak scaling analysis",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--hostname", default="any")


def run(args):
    """Run `fluidsim bench-analysis` command."""
    args = parse_args_dim(args)
    if len(args.input_dirs) == 0:
        args.input_dirs = [args.input_dir]

    fig = plt.figure(figsize=[12, 5])
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)
    t_min = 1e10
    for in_dir in args.input_dirs:
        fig, t_min = plot_scaling(
            in_dir,
            args.solver,
            args.hostname,
            args.n0,
            args.n1,
            args.n2,
            show=True,
            type_plot=args.type_plot,
            fig=fig,
            ax0=ax0,
            ax1=ax1,
            t_min_cum=t_min,
        )

    if args.save:
        figname = "fig_bench_" + os.path.basename(args.input_dir) + ".png"
        fig.savefig(figname)
    else:
        plt.show()
