import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.forcing.kolmogorov import (
    extend_simul_class,
    KolmogorovFlow,
    KolmogorovFlowNormalized,
)

from fluidsim.util.testing import classproperty
from fluidsim.solvers.ns3d.test_solver import TestSimulBase


class TestKolmo(TestSimulBase):
    @staticmethod
    def _init_grid(params, nx):
        params.oper.nx = params.oper.ny = nx
        try:
            params.oper.nz = nx
        except AttributeError:
            pass

    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns3d.strat.solver import Simul as SimulNotExtended

        return extend_simul_class(
            SimulNotExtended, [KolmogorovFlow, KolmogorovFlowNormalized]
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

        params.forcing.enable = True
        params.forcing.type = "kolmogorov_flow"
        params.forcing.kolmo.ik = 3
        params.forcing.kolmo.amplitude = 2.0

        return params

    def test_kolmo(self):
        sim = self.sim
        sim.time_stepping.start()
        if mpi.rank == 0:
            self.check_results(sim)

    def check_results(self, sim):
        data = sim.output.spatial_means.load()
        PK_tot = data["PK_tot"]
        PK2 = data["PK2"]
        dt = sim.params.time_stepping.deltat_max
        PK2_theo = sim.params.forcing.kolmo.amplitude**2 * dt / 2 / 2
        assert np.allclose(PK2, PK2_theo)
        PK_tot_theo = PK2_theo + np.array([0, 2, 4]) * dt
        assert np.allclose(PK_tot, PK_tot_theo)


class TestKolmo2D(TestKolmo):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns2d.solver import Simul as SimulNotExtended

        return extend_simul_class(
            SimulNotExtended, [KolmogorovFlow, KolmogorovFlowNormalized]
        )

    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.key_forced = "ux_fft"
        params.forcing.kolmo.letter_gradient = "y"
        return params


class TestKolmoNormalized(TestKolmo):
    nx = 24

    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.type = "kolmogorov_flow_normalized"
        params.forcing.nkmax_forcing = 6

        params.forcing.kolmo.amplitude = None
        params.forcing.forcing_rate = 10.0

    def check_results(self, sim):
        data = sim.output.spatial_means.load()
        PK_tot = data["PK_tot"]
        assert np.allclose(PK_tot, sim.params.forcing.forcing_rate)


class TestKolmoNormalized2D(TestKolmoNormalized):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns2d.strat.solver import Simul as SimulNotExtended

        return extend_simul_class(
            SimulNotExtended, [KolmogorovFlow, KolmogorovFlowNormalized]
        )
