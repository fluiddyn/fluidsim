"""Main script for fluidsim command (:mod:`fluidsim.util.console.__main__`)
===========================================================================

"""
import argparse
from fluidsim import __version__, get_local_version
from . import bench, bench_analysis, profile
from .util import ConsoleError


def compute_name_command(module):
    name = module.__name__.split(".")[-1]
    return name.replace("_", "-")


def add_subparser(subparsers, module, description=None, subcommand=None):
    """Add a subparser for a module. Expects two functions `init_parser` and
    `run` to be defined within the module.

    """
    if subcommand is None:
        subcommand = compute_name_command(module)

    parser_subcommand = subparsers.add_parser(subcommand, description=description)
    try:
        module.init_parser(parser_subcommand)
    except ConsoleError:
        return

    parser_subcommand.set_defaults(func=module.run)


def print_verbose_version(args):
    """Prints verbose version including local SCM version if it exists."""
    print("fluidsim", __version__)
    local_version = get_local_version()
    if local_version != __version__:
        print("local version:", get_local_version())


def get_parser():
    """Defines parser for command `fluidsim` and all its sub-commands."""
    parser = argparse.ArgumentParser(
        prog="fluidsim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Console utilities for FluidSim",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        help='see "fluidsim {subcommand} -h" for more details'
    )
    for module in (bench, bench_analysis, profile):
        add_subparser(subparsers, module, module.description)

    parser_version = subparsers.add_parser(
        "version", description=print_verbose_version.__doc__
    )
    parser_version.set_defaults(func=print_verbose_version)

    return parser


def run():
    """Parse arguments and execute the current script."""
    parser = get_parser()

    def print_help(args):
        """To print help when no subcommands are given."""
        parser.print_help()

    parser.set_defaults(func=print_help)
    args = parser.parse_args()
    args.func(args)


def _run_from_module(module):
    parser = argparse.ArgumentParser(
        prog="fluidsim" + compute_name_command(module),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=module.description,
    )

    module.init_parser(parser)
    args = parser.parse_args()
    module.run(args)


def run_profile():
    _run_from_module(profile)


def run_bench():
    _run_from_module(bench)


def run_bench_analysis():
    _run_from_module(bench_analysis)


if __name__ == "__main__":
    run()
