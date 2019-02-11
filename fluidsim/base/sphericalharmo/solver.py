"""Base solver (:mod:`fluidsim.base.solvers.sphericalharmo`)
============================================================

.. autoclass:: InfoSolverSphericalHarmo
   :members:
   :private-members:

.. autoclass:: SimulSphericalHarmo
   :members:
   :private-members:


"""

from ..solvers.pseudo_spect import (
    InfoSolverPseudoSpectral,
    SimulBasePseudoSpectral,
)


class InfoSolverSphericalHarmo(InfoSolverPseudoSpectral):
    """Contain the information on a base pseudo-spectral solver."""

    def _init_root(self):
        """Init. `self` by writting the information on the solver.

        The first-level classes for this base solver are the same as
        for the 2D pseudo-spectral base solver except the class:

        - :class:`fluidsim.operators.operators.OperatorsPseudoSpectral2D`

        """

        super()._init_root()

        here = "fluidsim.base.sphericalharmo"

        self.module_name = here + ".solver"
        self.class_name = "SimulSphericalHarmo"
        self.short_name = "BaseSH"

        self.classes.Operators.module_name = "fluidsim.operators.sphericalharmo"
        self.classes.Operators.class_name = "OperatorsSphericalHarmonics"

        self.classes.State.module_name = here + ".state"
        self.classes.State.class_name = "StateSphericalHarmo"

        self.classes.Output.module_name = here + ".output"
        self.classes.Output.class_name = "Output"


class SimulSphericalHarmo(SimulBasePseudoSpectral):
    """Pseudo-spectral base solver."""

    InfoSolver = InfoSolverSphericalHarmo


Simul = SimulSphericalHarmo

if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = "test"
    params.time_stepping.USE_CFL = False
    params.time_stepping.t_end = 2.0
    params.time_stepping.deltat0 = 0.1

    sim = Simul(params)
    sim.time_stepping.start()
