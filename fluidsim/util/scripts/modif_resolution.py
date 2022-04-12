"""
Create a new file with another resolution (:mod:`fluidsim.util.scripts.modif_resolution`)
=========================================================================================

.. autofunction:: create_parser

.. autofunction:: main

"""

import argparse
import sys

from fluiddyn.util import mpi

from fluidsim.util import modif_resolution_from_dir_memory_efficient

from fluidsim.util.scripts import parse_args


def create_parser():
    """Create the argument parser with default arguments"""
    parser = argparse.ArgumentParser(
        description="Create a new state file with a different resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        type=str,
        help="Input: path towards a directory or a file",
    )

    parser.add_argument(
        "coef_modif_resol",
        type=str,
        help="Input: coefficient of resolution change",
    )

    parser.add_argument(
        "--t_approx",
        type=float,
        default=None,
        help="Approximate time to choose the file from which to restart",
    )

    return parser


def main(args=None, **defaults):
    """Entry point fluidsim-modif-resolution"""
    parser = create_parser()

    if defaults:
        parser.set_defaults(**defaults)

    args = parse_args(parser, args)

    modif_resolution_from_dir_memory_efficient(
        args.path, args.t_approx, eval(args.coef_modif_resol)
    )


if "sphinx" in sys.modules:
    from textwrap import indent
    from unittest.mock import patch

    with patch.object(sys, "argv", ["fluidsim-modif-resol"]):
        parser = create_parser()

    __doc__ += """
Help message
------------

.. code-block::

""" + indent(
        parser.format_help(), "    "
    )
