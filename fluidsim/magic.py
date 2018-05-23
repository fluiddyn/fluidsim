"""IPython / Jupyter magic commands
===================================

.. autoclass:: FluidsimMagics
   :members:
   :private-members:

"""
from __future__ import print_function

from IPython.core.magic import Magics, magics_class, line_magic, line_cell_magic
from IPython.core import magic_arguments

from pprint import pprint
from fluidsim.util.util import (
    available_solver_keys,
    import_simul_class_from_key,
    load_sim_for_plot,
    load_state_phys_file,
)


@magics_class
class FluidsimMagics(Magics):
    """Magics simplifies the instantiation steps for a Simul object. It
    creates `params` and `sim` as a variables within the magic, unless
    explicitly demanded to be shared in the user namespace.

    It can be loaded in IPython or Jupyter as

    >>> %load_ext fluidsim.magic

    Examples
    --------
    List all available solvers
    >>> %fluidsim

    Create default parameters for a solver as variable `params`.
    >>> %fluidsim ns2d

    Create default parameters available in the user namespace as well.
    >>> %fluidsim ns2d -u

    Instantiate simulation with default parameters or with a preexisting
    `params` instance in the current namespace.
    >>> %%fluidsim ns2d
    ... sim.time_stepping_start()

    Modify parameters.
    >>> %%fluidsim
    ... params.forcing.enabled = True

    Execute code with `sim`.
    >>> %%fluidsim
    ... sim.info

    Load existing simulation excluding state_phys files.
    >>> %fluidsim_load

    Load existing simulation all options: user namespace, with state_phys
    files, merging parameters
    >>> %fluidsim_load -u -a -m

    Reset `sim` and `params`.
    >>> %fluidsim_reset

    Quick reference
    >>> %fluidsim_help

    """

    def __init__(self, shell):
        """ Init the FluidsimMagic with an empty Simul. """
        super(FluidsimMagics, self).__init__(shell)
        self.sim = None
        self.params = None
        self.Simul = None

    def init_params(self, line):
        if self.params is None:
            self.Simul = Simul = import_simul_class_from_key(line)
            self.params = Simul.create_default_params()
            print("Created default parameters for", line, "-> params")

    def init_simul(self, line):
        if self.sim is None:
            self.sim = self.Simul(self.params)
        else:
            print("Using existing", type(self.sim), "instance -> sim")

    def is_inexistent(self, varname):
        user_ns = self.shell.user_ns
        return varname not in user_ns or user_ns[varname] is None

    def populate_userns(self):
        if self.is_inexistent("sim"):
            self.shell.user_ns["sim"] = self.sim
        if self.is_inexistent("params"):
            self.shell.user_ns["params"] = self.params

    def unpopulate_userns(self):
        if not self.is_inexistent("sim"):
            del self.shell.user_ns["sim"]

        if not self.is_inexistent("params"):
            del self.shell.user_ns["params"]

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("solver", nargs="?", default="")
    @magic_arguments.argument("-u", "--user-namespace", action="store_true")
    @line_cell_magic
    def fluidsim(self, line, cell=None):
        args = magic_arguments.parse_argstring(self.fluidsim, line)

        if args.solver == "" and cell is None:
            print("Available solvers:")
            pprint(available_solver_keys(), indent=4, compact=True)
            print("\nInitialized:")
            pprint(dict(params=type(self.params), sim=type(self.sim)), indent=4)
        elif args.solver != "" and cell is None:
            self.init_params(args.solver)
        elif cell is not None:
            self.init_params(args.solver)
            params = self.params
            if "sim" in cell:
                self.init_simul(args.solver)
                sim = self.sim

        if args.user_namespace:
            self.populate_userns()

        if cell is not None:
            try:
                return eval(cell)
            except SyntaxError:
                # To handle assignment statements
                exec(cell, self.shell.user_ns, locals())

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("-u", "--user-namespace", action="store_true")
    @line_magic
    def fluidsim_reset(self, line):
        self.sim = None
        self.params = None

        args = magic_arguments.parse_argstring(self.fluidsim_reset, line)
        if args.user_namespace:
            self.unpopulate_userns()

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("-u", "--user-namespace", action="store_true")
    @magic_arguments.argument("-s", "--state-phys", action="store_true")
    @magic_arguments.argument("-m", "--merge-missing-params", action="store_true")
    @line_magic
    def fluidsim_load(self, line):
        args = magic_arguments.parse_argstring(self.fluidsim_load, line)
        if args.state_phys:
            self.sim = load_state_phys_file(None, args.merge_missing_params)
        else:
            self.sim = load_sim_for_plot(None, args.merge_missing_params)
        self.params = self.sim.params
        self.Simul = type(self.sim)
        if args.user_namespace:
            self.populate_userns()

    @line_magic
    def fluidsim_help(self, line):
        print(4 * ' ' + self.__doc__)

def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(FluidsimMagics)
