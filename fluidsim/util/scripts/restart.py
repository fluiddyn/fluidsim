"""Restart a fluidsim simulation (:mod:`fluidsim.util.scripts.restart`)
=======================================================================

.. autofunction:: create_parser

.. autofunction:: restart

.. autofunction:: main

"""

import sys

import h5py

from fluiddyn.util import mpi

from fluidsim_core.scripts.restart import RestarterABC

from fluidsim.base.output.phys_fields import time_from_path
from fluidsim import load_for_restart


class Restarter(RestarterABC):
    def _get_params_simul_class(self, args):
        return load_for_restart(
            args.path, args.t_approx, args.merge_missing_params
        )

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
main = _restarter.restart


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
