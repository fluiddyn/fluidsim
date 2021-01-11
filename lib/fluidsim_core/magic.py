"""IPython / Jupyter magic commands

.. autoclass:: MagicsCore
   :members:
   :private-members:

"""
from pprint import pprint

from fluiddyn.io.query import query_yes_no
from IPython.core.magic import Magics, line_magic
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
)

from .loader import available_solvers, import_cls_simul


class MagicsCore(Magics):
    """Magics simplifies the instantiation steps for a Simul object.
    A class variable ``entrypoint_grp`` informs which modules to import from.
    See also :mod:`fluidsim_core.loader`.

    """

    entrypoint_grp = None

    def is_defined(self, varname):
        user_ns = self.shell.user_ns
        return varname in user_ns and user_ns[varname] is not None

    @magic_arguments()
    @argument("solver", nargs="?", default="")
    @argument("-f", "--force-overwrite", action="store_true")
    @line_magic
    def fluidsim(self, line):
        args = parse_argstring(self.fluidsim, line)

        if args.solver == "":
            print("Available solvers:")
            pprint(available_solvers(self.entrypoint_grp), indent=4, compact=True)
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
        Simul = import_cls_simul(args.solver, self.entrypoint_grp)
        params = Simul.create_default_params()
        print(
            "Created Simul class and default parameters for",
            line,
            "-> Simul, params",
        )
        user_ns["Simul"] = Simul
        user_ns["params"] = params

    @magic_arguments()
    @line_magic
    def fluidsim_reset(self, line):
        for varname in ("sim", "params", "Simul"):
            if self.is_defined(varname):
                del self.shell.user_ns[varname]

    @line_magic
    def fluidsim_help(self, line):
        print(4 * " " + self.__doc__)
