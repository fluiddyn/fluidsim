"""Main script for fluidsim command (:mod:`fluidsim.util.console.__main__`)
===========================================================================

"""
import argparse
from ..._version import __version__
from . import bench, bench_analysis
from .util import MyValueError


def add_subparser(subparsers, module, description=None, subcommand=None):
    """Add a subparser for a module. Expects two functions `init_parser` and
    `run` to be defined within the module.

    """
    if subcommand is None:
        subcommand = module.__name__.split('.')[-1]
        subcommand = subcommand.replace('_', '-')

    parser_subcommand = subparsers.add_parser(
        subcommand, description=description)
    try:
        module.init_parser(parser_subcommand)
    except MyValueError:
        return

    parser_subcommand.set_defaults(func=module.run)


def get_parser():
    """Defines parser for command `fluidsim` and all its sub-commands."""
    parser = argparse.ArgumentParser(
        prog='fluidsim',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Console utilities for FluidSim')
    parser.add_argument('-V', '--version',
                        action='version',
                        version=__version__)

    subparsers = parser.add_subparsers(
        help='see "fluidsim {subcommand} -h" for more details')
    for module, description in [
            (bench, 'Run benchmarks of FluidSim solvers'),
            (bench_analysis, 'Plot results of benchmarks')]:
        add_subparser(subparsers, module, description)

    return parser


def run():
    """Parse arguments and execute the current script."""
    parser = get_parser()
    def print_help(args):
        parser.print_help()

    parser.set_defaults(func=print_help)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    run()
