"""Load and plot benchmarks (:mod:`fluidsim.util.console.bench_analysis`)
=========================================================================

"""
import os
from glob import glob
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .bench import (
    path_results, parse_args_dim, init_parser_base)


def load_bench(path_dir, solver, hostname='any'):
    """Load benchmarks results saved as JSON files."""

    dicts = []
    for path in glob(path_dir + '/result_bench_{}*.json'.format(solver)):
        with open(path) as f:
            d = json.load(f)

        if hostname != 'any' and not d['hostname'].startswith(hostname):
            continue

        dicts.append(d)

    df = pd.DataFrame(dicts)
    df = df[df.columns.difference(['pid', 'time_as_str'])]
    return df


def filter_by_shape(df, n0, n1):
    """Filters all results with the same `n0` and same `n1`."""

    df = df[(df.n0 == n0) & (df.n1 == n1)]
    return df[df.columns.difference(['n0', 'n1'])]


def filter_by_shapeloc(df, n0_loc, n1_loc):
    """Filters all results with the same `n0_loc * n1_loc`.

    This implies shapeK_loc has same no. of points. This is a weak check and
    the shapeK_loc may not have the same shape. Make sure all shapes are
    estimated and tested apriori using:

    >>> fluidsim bench -e
    >>> mpirun -np {nb_proc} fluidsim bench -c

    """
    df = df[(df.n0_loc * df.n1_loc == n0_loc * n1_loc)]
    return df[df.columns.difference(['n0_loc', 'n1_loc'])]


def plot_scaling(
        path_dir, solver, hostname, n0, n1, show=True,
        type_time='usr', type_plot='strong', fig=None, ax0=None, ax1=None,
        name_dir=None):
    """Plot speedup vs number of processes from benchmark results."""

    def check_empty(df):
        """Check if the dataframe is empty."""
        if df.empty:
            raise ValueError(
                'No benchmarks corresponding to the input parameters')

    if name_dir is None:
        name_dir = os.path.basename(
            os.path.abspath(path_dir)).replace('_', ' ').upper()

    df = load_bench(path_dir, solver, hostname)
    check_empty(df)

    df.t_elapsed_sys /= df.nb_iter
    df.t_elapsed_usr /= df.nb_iter

    if type_plot == 'strong':
        df_filter = filter_by_shape(df, n0, n1)
    elif type_plot == 'weak':
        df_filter = filter_by_shapeloc(df, n0, n1)
    else:
        raise ValueError('Unknown plot type.')

    def group_df(df):
        """Group and take median dataframe results with same number of processes."""
        # for "scaling" (mpi)
        df = df[df.nb_proc > 1]
        check_empty(df)
        if show:
            print(df)

        nb_proc_min = df.nb_proc.min()
        df_grouped = df.groupby(['key_solver', 'nb_proc']).quantile(q=0.2)
        return df_grouped, nb_proc_min

    df_filter, nb_proc_min_filter = group_df(df_filter)
    keys_filter = [
        k for k in df_filter.columns if k.startswith('t_elapsed_' + type_time)]

    df_filter = df_filter[keys_filter]
    df_filter_nb_proc_min = df_filter.xs(nb_proc_min_filter, level=1)

    def get_min(df):
        """Get minima from a set of results."""
        m = df.as_matrix()
        i0, i1 = np.unravel_index(np.argmin(m), m.shape)
        mymin = m[i0, i1]
        ind = df.index[i0]
        key = df.columns[i1]
        return mymin, ind, key

    t_min_filter, name_min_filter, key_min_filter = get_min(
        df_filter_nb_proc_min)

    # Finally, start preparing figure and plot
    if fig is None or ax0 is None or ax1 is None:
        fig, axes = plt.subplots(1, 2)
        ax0, ax1 = axes.ravel()

    ax0.set_ylabel('speedup ({} scaling)'.format(type_plot))
    ax1.set_ylabel('efficiency % ({} scaling)'.format(type_plot))

    def plot_once(ax, x, y, label, linestyle='-k'):
        """Avoid plotting the same label again."""
        plotted = any(
            [lines.get_label() == label for lines in ax.get_lines()])
        if not plotted:
            ax.plot(x, y, linestyle, label=label)

    def add_hline(ax, series):
        """Add a horizontal line to the axis."""
        y = series.iloc[0]
        xmin = series.index.min()
        xmax = series.index.max()
        plot_once(ax, [xmin, xmax], [y, y], 'linear')

    def set_label_scale(ax, yscale='log'):
        """Set log scale, legend and label of the axis."""
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_xlabel('number of processes')

    # Plot speedup
    for name in df_filter.index.levels[0]:
        tmp = df_filter.loc[name]
        # print(name)

        for k in keys_filter:
            speedup = t_min_filter / tmp[k] * nb_proc_min_filter
            ax0.plot(
                speedup.index, speedup.values, 'x-',
                label='{}, {}'.format(name, name_dir))

    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    if type_plot == 'strong':
        theoretical = [speedup.index.min(), speedup.index.max()]
        plot_once(ax0, theoretical, theoretical, 'linear')
    else:
        add_hline(ax0, speedup)

    set_label_scale(ax0)
    ax0.legend()
    ax0.set_title('Best for {} processes: {}, {}={:.2f} ms'.format(
        nb_proc_min_filter, name_min_filter, key_min_filter,
        t_min_filter * 1000))

    # Plot efficiency
    for name in df_filter.index.levels[0]:
        for k in keys_filter:
            speedup = t_min_filter / tmp[k] * nb_proc_min_filter
            if type_plot == 'strong':
                efficiency = speedup / speedup.index * 100
            else:
                efficiency = speedup / speedup.iloc[0] * 100
            add_hline(ax1, efficiency)
            ax1.plot(
                efficiency.index, efficiency.values, 'x-',
                label='{}, {}'.format(name, name_dir))

    set_label_scale(ax1, 'linear')

    if show:
        plt.show()
    return fig


def init_parser(parser):
    """Initialize argument parser for `fluidsim bench-analysis`."""

    init_parser_base(parser)
    parser.add_argument(
        '-i', '--input-dir', default=path_results,
        help='plot results from a single directory')
    parser.add_argument(
        '--input-dirs',
        nargs='+',
        default=[],
        help='plot results from mulitiple input directories')
    parser.add_argument(
        '-p', '--type-plot', default='strong',
        help='load and plot data for strong or weak scaling analysis')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--hostname', default='any')


def run(args):
    """Run `fluidsim bench-analysis` command."""
    args = parse_args_dim(args)
    if args.dim == '3d':
        raise NotImplementedError
    else:
        if len(args.input_dirs) == 0:
            args.input_dirs = [args.input_dir]

        fig = plt.figure(figsize=[12, 5])
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)
        for in_dir in args.input_dirs:
            fig = plot_scaling(
                in_dir, args.solver, args.hostname, args.n0, args.n1,
                show=False, type_plot=args.type_plot, fig=fig, ax0=ax0, ax1=ax1)

        if args.save:
            figname = 'fig_bench_' + os.path.basename(args.input_dir) + '.png'
            fig.savefig(figname)
        else:
            plt.show()
