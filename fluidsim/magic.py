"""IPython / Jupyter magic commands
===================================

.. autoclass:: FluidsimMagics
   :members:
   :private-members:

"""
from fluiddyn.io.query import query_yes_no
from fluidsim_core.magic import MagicsCore
from IPython.core.magic import line_magic, magics_class
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
)

from fluidsim import load_sim_for_plot, load_state_phys_file


@magics_class
class FluidsimMagics(MagicsCore):
    """Magic commands can be loaded in IPython or Jupyter as

    >>> %load_ext fluidsim.magic

    Examples
    --------

    - Magic command %fluidsim

    %fluidsim creates the variables `params` and `Simul` for a particular solver.

    Create default parameters for a solver:

    >>> %fluidsim ns2d

    If a variable `params` already exists, you will be ask if you really want to
    overwrite it. To skip this question:

    >>> %fluidsim ns2d -f

    List all available solvers and initialized simulation:

    >>> %fluidsim

    - Magic command %fluidsim_load

    %fluidsim_load creates the variables `sim`, `params` and `Simul` from an
    existing simulation.

    Load existing simulation excluding state_phys files:

    >>> %fluidsim_load

    Load existing simulation all options: force overwrite, with state_phys
    files, merging parameters:

    >>> %fluidsim_load -f -s -t -m

    - Other fluidsim magic commands

    Quick reference (print this help message):

    >>> %fluidsim_help

    Delete the objects `sim` and `params`:

    >>> %fluidsim_reset

    """

    entrypoint_grp = "fluidsim.solvers"

    @magic_arguments()
    @argument("-f", "--force-overwrite", action="store_true")
    @argument("-s", "--state-phys", action="store_true")
    @argument("-t", "--t-approx", type=float, default=None)
    @argument("-m", "--merge-missing-params", action="store_true")
    @argument(
        "directory",
        nargs="?",
        help="Optional: absolute path/relative path/name of directory",
        default=None,
    )
    @line_magic
    def fluidsim_load(self, line):
        args = parse_argstring(self.fluidsim_load, line)

        if not args.force_overwrite and (self.is_defined("sim")):
            if not query_yes_no(
                "The variables `sim` is defined in your user namespace. "
                "Do you really want to overwrite it?"
            ):
                return

        if args.state_phys:
            sim = load_state_phys_file(
                name_dir=args.directory,
                t_approx=args.t_approx,
                merge_missing_params=args.merge_missing_params,
            )
        else:
            sim = load_sim_for_plot(
                name_dir=args.directory,
                merge_missing_params=args.merge_missing_params,
            )

        user_ns = self.shell.user_ns
        user_ns["Simul"] = type(sim)
        user_ns["params"] = sim.params
        user_ns["sim"] = sim


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    fluidsim_magics = FluidsimMagics(ipython)
    ipython.register_magics(fluidsim_magics)
