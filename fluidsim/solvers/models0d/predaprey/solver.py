"""Predator-prey solver (:mod:`fluidsim.solvers.models0d.predaprey.solver`)
===========================================================================

This module provides classes to solve the Lotka-Volterra equations.

.. autoclass:: InfoSolverPredaPrey
   :members:
   :private-members:

.. autoclass:: Simul
   :members:
   :private-members:

.. autoclass:: StatePredaPrey
   :members:
   :private-members:

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.base import InfoSolverBase, SimulBase
from fluidsim.base.state import StateBase


class StatePredaPrey(StateBase):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        info_solver.classes.State._set_attribs(
            {
                "keys_state_phys": ["X", "Y"],
                "keys_computable": [],
                "keys_phys_needed": ["X", "Y"],
                "keys_linear_eigenmodes": ["X", "Y"],
            }
        )


class InfoSolverPredaPrey(InfoSolverBase):
    """Contain the information on the solver predaprey."""

    def _init_root(self):
        super()._init_root()

        package = "fluidsim.solvers.models0d.predaprey"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "predaprey"

        classes = self.classes

        classes.State.module_name = package + ".solver"
        classes.State.class_name = "StatePredaPrey"

        classes.Output.module_name = package + ".output"
        classes.Output.class_name = "Output"


class Simul(SimulBase):
    """Solve the Lotka-Volterra equations."""

    InfoSolver = InfoSolverPredaPrey

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method)."""
        SimulBase._complete_params_with_default(params)
        attribs = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
        params._set_attribs(attribs)

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        p = self.params
        self.Xs = p.C / p.D
        self.Ys = p.A / p.B

    def tendencies_nonlin(self, state=None, old=None):
        r"""Compute the nonlinear tendencies.

        Parameters
        ----------

        state : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state.  If `state is not None`, the variables
            are computed from it, otherwise, they are taken from the global
            state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the variables.

        Notes
        -----

        The Lotka-Volterra equations can be written

        .. math::
           \dot X = AX - B XY,
           \dot Y = -CY + D XY.


        """
        p = self.params

        if state is None:
            state = self.state.state_phys

        X = state.get_var("X")
        Y = state.get_var("Y")

        if old is None:
            tendencies = SetOfVariables(like=self.state.state_phys)
        else:
            tendencies = old
        tendencies.set_var("X", p.A * X - p.B * X * Y)
        tendencies.set_var("Y", -p.C * Y + p.D * X * Y)

        if self.params.forcing.enable:
            tendencies += self.forcing.get_forcing()

        return tendencies


if __name__ == "__main__":
    import fluiddyn as fld

    params = Simul.create_default_params()

    params.time_stepping.deltat0 = 0.1
    params.time_stepping.t_end = 20

    params.output.periods_print.print_stdout = 0.01

    sim = Simul(params)

    sim.state.state_phys.set_var("X", 2)
    sim.state.state_phys.set_var("Y", 1.1)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    fld.show()
