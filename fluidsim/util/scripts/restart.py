"""Restart a fluidsim simulation (:mod:`fluidsim.util.scripts.restart`)
=======================================================================

.. autofunction:: create_parser

.. autofunction:: restart

.. autofunction:: main

.. autoclass:: Restarter
   :members:
   :private-members:

"""

import sys

import h5py

from fluiddyn.util import mpi

from fluidsim_core.scripts.restart import RestarterABC

from fluidsim.base.output.phys_fields import time_from_path
from fluidsim import load_for_restart


class Restarter(RestarterABC):
    def create_parser(self):
        parser = super().create_parser()

        parser.add_argument(
            "--t_approx",
            type=float,
            default=None,
            help="Approximate time to choose the file from which to restart",
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
            "--merge-missing-params",
            action="store_true",
            help=(
                "Can be used to load old simulations carried out with "
                "an old fluidsim version."
            ),
        )
        parser.add_argument(
            "--max-elapsed",
            type=str,
            default=None,
            help="Maximum elapsed time.",
        )

        return parser

    def _get_params_simul_class(self, args):
        return load_for_restart(
            args.path, args.t_approx, args.merge_missing_params
        )

    def _set_params_time_stepping(self, params, args):
        if args.add_to_t_end is not None:
            params.time_stepping.t_end += args.add_to_t_end
        if args.add_to_it_end is not None:
            params.time_stepping.it_end += args.add_to_it_end
        if args.t_end is not None:
            params.time_stepping.t_end = args.t_end
        if args.it_end is not None:
            params.time_stepping.it_end += args.it_end

    def _start_sim(self, sim, args):
        sim.time_stepping.start()

    def _check_params_time_stepping(self, params, path_file, args):
        if params.time_stepping.USE_T_END:
            if params.time_stepping.t_end <= time_from_path(path_file):
                mpi.printby0(
                    f"{params.time_stepping.t_end = } <= {time_from_path(path_file) = }"
                )
                raise ValueError
        else:
            with h5py.File(path_file, "r") as file:
                it_file = file["/state_phys"].attrs["it"]
            if params.time_stepping.it_end <= it_file:
                mpi.printby0(f"{params.time_stepping.it_end = } <= {it_file = }")
                raise ValueError


_restarter = Restarter()

create_parser = _restarter.create_parser
restart = _restarter.restart
main = _restarter.main


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
