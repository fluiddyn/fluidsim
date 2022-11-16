"""Utilities for fluidsim scripts

Provides:

.. autosummary::
   :toctree:

    restart

.. autofunction:: parse_args

"""
import shlex

from fluiddyn.util import mpi


def parse_args(parser, args=None):
    """Parse the arguments"""
    if args is not None and isinstance(args, str):
        args = shlex.split(args)
    args = parser.parse_args(args)
    mpi.printby0(args)
    return args
