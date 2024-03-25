from fluiddyn.util import mpi

from fluidsim.util.testing import classproperty
from fluidsim.solvers.ns3d.test_solver import TestSimulBase

from fluidsim.base.turb_model import extend_simul_class, SmagorinskyModel
from fluidsim.base.output.horiz_means import HorizontalMeans


class TestSmagorinsky(TestSimulBase):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns3d.solver import Simul as SimulNotExtended

        return extend_simul_class(
            SimulNotExtended, [SmagorinskyModel, HorizontalMeans]
        )

    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.init_fields.noise.velo_max = 1e-12
        params.nu_2 = 0.0
        params.nu_4 = 0.0
        params.nu_8 = 0.0

        dt = params.time_stepping.deltat_max = 1e-2
        params.time_stepping.t_end = 1.5 * dt
        params.output.periods_print.print_stdout = dt
        params.output.periods_save.spatial_means = dt
        params.output.periods_save.horiz_means = dt

        params.turb_model.enable = True
        params.turb_model.type = "smagorinsky"
        params.turb_model.smagorinsky.C = 0.18

    def test_smagorinsky(self):
        sim = self.sim
        sim.info_solver.classes.TurbModel.classes
        sim.turb_model
        sim.time_stepping.start()

        if mpi.nb_proc > 1:
            return

        sim.output.horiz_means.plot()
