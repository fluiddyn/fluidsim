"""Restart a fluidsim simulation (:mod:`fluidsim.util.scripts.restart`)
=======================================================================

.. autofunction:: create_parser

.. autofunction:: restart

.. autofunction:: main

"""

import argparse
import sys
from pathlib import Path

# import potentially needed for the exec
from math import pi

import h5py

from fluiddyn.util import mpi

from fluidsim.base.output.phys_fields import time_from_path
from fluidsim.util.util import load_for_restart
from fluidsim.util.scripts import parse_args


def create_parser():
    """Create the argument parser with default arguments"""
    parser = argparse.ArgumentParser(
        description="Restart a fluidsim simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    parser.add_argument(
        "--modify-params",
        type=str,
        default=None,
        help="Code modifying the `params` object.",
    )

    return parser


def restart(args=None, **defaults):
    """Restart a simulation"""
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

    if args.modify_params is not None:
        exec(args.modify_params)

    if args.only_check:
        mpi.printby0(params)
        return params, None

    if params.time_stepping.USE_T_END:
        if params.time_stepping.t_end <= time_from_path(path_file):
            mpi.printby0(
                f"{params.time_stepping.t_end = } <= {time_from_path(path_file) = }"
            )
            return params, None
    else:
        with h5py.File(path_file, "r") as file:
            it_file = file["/state_phys"].attrs["it"]
        if params.time_stepping.it_end <= it_file:
            mpi.printby0(f"{params.time_stepping.it_end = } <= {it_file = }")
            return params, None

    sim = Simul(params)

    if args.only_init:
        return params, sim

    sim.time_stepping.start()

    mpi.printby0(
        f"""
# To visualize with IPython:

cd {sim.output.path_run}
ipython --matplotlib -i -c "from fluidsim import load; sim = load()"
"""
    )

    return params, sim


def main():
    """Entry point fluidsim-restart"""
    restart()


if "sphinx" in sys.modules:
    from textwrap import indent
    from unittest.mock import patch

    with patch.object(sys, "argv", ["fluidsim-restart"]):
        parser = create_parser()

    __doc__ += """
Help message
------------

.. code-block::

""" + indent(
        parser.format_help(), "    "
    )
