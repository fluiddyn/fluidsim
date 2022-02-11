"""Basilisk solver (:mod:`fluidsim.base.basilisk.solver`)
=========================================================

Provides:

.. autoclass:: InfoSolverBasilisk
   :members:
   :private-members:

.. autoclass:: SimulBasilisk
   :members:
   :private-members:

"""

from fluidsim.base.solvers.base import SimulBase
from fluidsim.base.solvers.info_base import InfoSolverBase

import basilisk.stream as basilisk


class InfoSolverBasilisk(InfoSolverBase):
    """Contain the information on a Basilisk solver."""

    def _init_root(self):

        super()._init_root()

        mod = "fluidsim.base.basilisk"

        self.module_name = mod + ".solver"
        self.class_name = "SimulBasilisk"
        self.short_name = "basil"

        classes = self.classes

        classes.State.module_name = mod + ".state"
        classes.State.class_name = "StateBasilisk"

        classes.TimeStepping.module_name = mod + ".time_stepping"
        classes.TimeStepping.class_name = "TimeSteppingBasilisk"

        classes.Operators.module_name = mod + ".operators"
        classes.Operators.class_name = "OperatorsBasilisk2D"

        classes.Output.module_name = mod + ".output"
        classes.Output.class_name = "OutputBasilisk"


class SimulBasilisk(SimulBase):
    """A solver for Basilisk."""

    InfoSolver = InfoSolverBasilisk

    def __init__(self, params):
        """Initialize parameters, state fields, and event loop."""
        bas = self.basilisk = basilisk
        super().__init__(params)

        def init(i, t):
            bas.omega.f = bas.noise

        bas.event(init, t=0.0)


Simul = SimulBasilisk

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    params.oper.nx = 128

    params.time_stepping.deltat0 = 2.4
    params.output.periods_print.print_stdout = 1e-15

    sim = Simul(params)
    sim.time_stepping.start()

    sim.output.print_stdout.plot_energy()
    plt.show()
