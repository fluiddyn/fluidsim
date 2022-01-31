"""
...

"""

import argparse
import sys
from pathlib import Path
from math import pi

import h5py

from fluiddyn.util import mpi

from fluidsim.util.util import load_for_restart
from fluidsim.base.output.phys_fields import time_from_path


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path of the directory or file from which to restart",
    )
    parser.add_argument(
        "--t_approx",
        type=float,
        default=None,
        help="Approximate time to choose the file from which to restart",
    )

    parser.add_argument(
        "--merge-missing-params",
        action="store_true",
        help=(
            "Can be used to load old simulations carried out with "
            "an old fluidsim version."
        ),
    )

    parser.add_argument(
        "-oc",
        "--only-check",
        action="store_true",
        help="Only check what should be done",
    )

    parser.add_argument(
        "-oi",
        "--only-init",
        action="store_true",
        help="Only run initialization phase",
    )

    parser.add_argument(
        "--new-dir-results",
        action="store_true",
        help="Create a new directory for the new simulation",
    )

    parser.add_argument(
        "--add-to-t_end",
        type=float,
        default=None,
        help="Time added to params.time_stepping.t_end",
    )
    parser.add_argument(
        "--add-to-it_end",
        type=int,
        default=None,
        help="Number of steps added to params.time_stepping.it_end",
    )

    parser.add_argument(
        "--t_end",
        type=float,
        default=None,
        help="params.time_stepping.t_end",
    )
    parser.add_argument(
        "--it_end",
        type=int,
        default=None,
        help="params.time_stepping.it_end",
    )

    return parser


def parse_args(parser, args):
    args = parser.parse_args(args)
    mpi.printby0(args)
    return args


def restart(args=None, **defaults):
    parser = create_parser()

    if defaults:
        parser.set_defaults(**defaults)

    args = parse_args(parser, args)

    params, Simul = load_for_restart(
        args.path, args.t_approx, args.merge_missing_params
    )

    params.NEW_DIR_RESULTS = args.new_dir_results

    if args.add_to_t_end is not None:
        params.time_stepping.t_end += args.add_to_t_end

    if args.add_to_it_end is not None:
        params.time_stepping.it_end += args.add_to_it_end

    if args.t_end is not None:
        params.time_stepping.t_end = args.t_end

    if args.it_end is not None:
        params.time_stepping.it_end += args.it_end

    if args.only_check or args.only_init:
        params.output.HAS_TO_SAVE = False

    path_file = Path(params.init_fields.from_file.path)
    mpi.printby0(path_file)

    # TODO: add a mechanism to modify params as for...
    # params.output.periods_save.spatiotemporal_spectra = 2 * pi / (4 * params.N)
    ...

    if args.only_check:
        mpi.printby0(params)
        sys.exit()

    if params.time_stepping.USE_T_END:
        if params.time_stepping.t_end <= time_from_path(path_file):
            mpi.printby0(
                f"{params.time_stepping.t_end = } <= {time_from_path(path_file) = }"
            )
            sys.exit()
    else:
        with h5py.File(path_file, "r") as file:
            it_file = file["/state_phys"].attrs["it"]
        if params.time_stepping.it_end <= it_file:
            mpi.printby0(f"{params.time_stepping.it_end = } <= {it_file = }")
            sys.exit()

    sim = Simul(params)

    if args.only_init:
        sys.exit()

    sim.time_stepping.start()
    return params, sim


if __name__ == "__main__":
    restart()
