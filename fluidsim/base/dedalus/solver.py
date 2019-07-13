"""Dedalus solver (:mod:`fluidsim.base.dedalus.solver`)
=========================================================

Provides:

.. autoclass:: InfoSolverDedalus
   :members:
   :private-members:

.. autoclass:: SimulDedalus
   :members:
   :private-members:

"""


from fluidsim.base.solvers.base import SimulBase
from fluidsim.base.solvers.info_base import InfoSolverBase

from dedalus import public as dedalus


class InfoSolverDedalus(InfoSolverBase):
    """Contain the information on a Dedalus solver."""

    def _init_root(self):

        super()._init_root()

        mod = "fluidsim.base.dedalus"

        self.module_name = mod + ".solver"
        self.class_name = "SimulDedalus"
        self.short_name = "dedalus"

        classes = self.classes

        classes.State.module_name = mod + ".state"
        classes.State.class_name = "StateDedalus"

        classes.TimeStepping.module_name = mod + ".time_stepping"
        classes.TimeStepping.class_name = "TimeSteppingDedalus"

        classes.Operators.module_name = mod + ".operators"
        classes.Operators.class_name = "OperatorsDedalus2D"

        classes.Output.module_name = mod + ".output"
        classes.Output.class_name = "OutputDedalus"


class SimulDedalus(SimulBase):
    """A solver for Dedalus."""

    dedalus = dedalus

    InfoSolver = InfoSolverDedalus

    @staticmethod
    def _complete_params_with_default(params):
        """Complete the `params` container (static method)."""
        SimulBase._complete_params_with_default(params)
        attribs = {"prandtl": 1.0, "rayleigh": 1e6, "F": 1.0}
        params._set_attribs(attribs)

    def create_problem(self):
        params = self.params

        # 2D Boussinesq hydrodynamics
        problem = self.dedalus.IVP(
            self.oper.domain,
            variables=["p", "b", "vx", "vz", "dz_b", "dz_vx", "dz_vz"],
        )
        problem.meta["p", "b", "vx", "vz"]["z"]["dirichlet"] = True
        problem.parameters["P"] = (params.rayleigh * params.prandtl) ** (-1 / 2)
        problem.parameters["R"] = (params.rayleigh / params.prandtl) ** (-1 / 2)
        problem.parameters["F"] = params.F
        problem.add_equation("dx(vx) + dz_vz = 0")
        problem.add_equation(
            "dt(b) - P*(dx(dx(b)) + dz(dz_b)) - F*vz       = -(vx*dx(b) + vz*dz_b)"
        )
        problem.add_equation(
            "dt(vx) - R*(dx(dx(vx)) + dz(dz_vx)) + dx(p)     = -(vx*dx(vx) + vz*dz_vx)"
        )
        problem.add_equation(
            "dt(vz) - R*(dx(dx(vz)) + dz(dz_vz)) + dz(p) - b = -(vx*dx(vz) + vz*dz_vz)"
        )
        problem.add_equation("dz_b - dz(b) = 0")
        problem.add_equation("dz_vx - dz(vx) = 0")
        problem.add_equation("dz_vz - dz(vz) = 0")
        problem.add_bc("left(b) = 0")
        problem.add_bc("left(vx) = 0")
        problem.add_bc("left(vz) = 0")
        problem.add_bc("right(b) = 0")
        problem.add_bc("right(vx) = 0")
        problem.add_bc("right(vz) = 0", condition="(nx != 0)")
        problem.add_bc("right(p) = 0", condition="(nx == 0)")
        return problem

    def build_dedalus_solver(self):
        solver = self.problem.build_solver(self.dedalus.timesteppers.RK222)
        return solver

    def init_dedalus(self):
        self.problem = self.create_problem()

        if not self.params.ONLY_COARSE_OPER:
            self.dedalus_solver = self.build_dedalus_solver()


Simul = SimulDedalus

if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    params.oper.nx = 48

    params.time_stepping.t_end = 10
    params.time_stepping.USE_CFL = True
    params.time_stepping.deltat0 = 2.0

    params.output.periods_print.print_stdout = 1e-15
    params.output.periods_save.phys_fields = 0.5

    sim = Simul(params)
    sim.time_stepping.start()
