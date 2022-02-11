"""Time stepping (:mod:`fluidsim.base.dedalus.time_stepping`)
==============================================================

Provides:

.. autoclass:: TimeSteppingDedalus
   :members:
   :private-members:

"""

from dedalus.extras import flow_tools

from fluidsim.base.time_stepping.base import TimeSteppingBase0


class TimeSteppingDedalus(TimeSteppingBase0):
    """Time stepping class to handle Dedalus's event loop and FluidSim output."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        TimeSteppingBase0._complete_params_with_default(params)

    def __init__(self, sim):
        if sim.params.time_stepping.USE_CFL:
            self.init_dedalus_cfl(sim)

        super().__init__(sim)

    def init_dedalus_cfl(self, sim):
        initial_dt = sim.params.time_stepping.deltat0
        if sim.params.ONLY_COARSE_OPER:
            self.CFL = None
            return
        self.CFL = flow_tools.CFL(
            sim.dedalus_solver,
            initial_dt=initial_dt,
            cadence=10,
            safety=1,
            max_change=1.5,
            min_change=0.5,
            max_dt=0.125,
            threshold=0.05,
        )
        self.CFL.add_velocities(("vx", "vz"))

    def start(self):
        self.deltat = self.params.time_stepping.deltat0
        super().start()

    def compute_time_increment_CLF(self):
        if self.CFL is None:
            self.deltat = None
        else:
            self.deltat = self.CFL.compute_dt()

    def one_time_step_computation(self):
        self.deltat = self.sim.dedalus_solver.step(self.deltat)
