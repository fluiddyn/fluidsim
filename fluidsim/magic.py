"""IPython / Jupyter magic commands
===================================

.. autoclass:: FluidsimMagics
   :members:
   :private-members:

"""

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core import magic_arguments

from pprint import pprint

from fluiddyn.io.query import query_yes_no

from fluidsim.util.util import (
    available_solver_keys,
    import_simul_class_from_key,
    load_sim_for_plot,
    load_state_phys_file,
)


@magics_class
class FluidsimMagics(Magics):
    """Magics simplifies the instantiation steps for a Simul object.

    It can be loaded in IPython or Jupyter as

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

    def __init__(self, shell, package_solvers):
        super().__init__(shell)
        self.package = package_solvers

    def is_defined(self, varname):
        user_ns = self.shell.user_ns
        return varname in user_ns and user_ns[varname] is not None

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("solver", nargs="?", default="")
    @magic_arguments.argument("-f", "--force-overwrite", action="store_true")
    @line_magic
    def fluidsim(self, line):
        args = magic_arguments.parse_argstring(self.fluidsim, line)

        if args.solver == "":
            print("Available solvers:")
            pprint(available_solver_keys(self.package), indent=4, compact=True)

            if self.is_defined("params") and self.is_defined("sim"):
                print("\nInitialized:")
                user_ns = self.shell.user_ns
                pprint(
                    dict(
                        params=type(user_ns["params"]), Simul=type(user_ns["sim"])
                    ),
                    indent=4,
                )
            return

        if not args.force_overwrite and (
            self.is_defined("params") or self.is_defined("Simul")
        ):
            if not query_yes_no(
                "At least one of the variables `params` or `Simul` is defined"
                " in your user namespace. Do you really want to overwrite "
                "them?"
            ):
                return

        user_ns = self.shell.user_ns
        Simul = import_simul_class_from_key(args.solver, self.package)
        params = Simul.create_default_params()
        print(
            "Created Simul class and default parameters for",
            line,
            "-> Simul, params",
        )
        user_ns["Simul"] = Simul
        user_ns["params"] = params

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("-f", "--force-overwrite", action="store_true")
    @magic_arguments.argument("-s", "--state-phys", action="store_true")
    @magic_arguments.argument("-t", "--t-approx", type=float, default=None)
    @magic_arguments.argument("-m", "--merge-missing-params", action="store_true")
    @magic_arguments.argument(
        "directory",
        nargs="?",
        help="Optional: absolute path/relative path/name of directory",
        default=None,
    )
    @line_magic
    def fluidsim_load(self, line):
        args = magic_arguments.parse_argstring(self.fluidsim_load, line)

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

    @magic_arguments.magic_arguments()
    @line_magic
    def fluidsim_reset(self, line):
        for varname in ("sim", "params", "Simul"):
            if self.is_defined(varname):
                del self.shell.user_ns[varname]

    @line_magic
    def fluidsim_help(self, line):
        print(4 * " " + self.__doc__)


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    fluidsim_magics = FluidsimMagics(ipython, "fluidsim.solvers")
    ipython.register_magics(fluidsim_magics)
