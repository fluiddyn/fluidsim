from fluidsim.extend_simul import extend_simul_class
from fluidsim.extend_simul.kolmogorov import (
    KolmogorovFlow,
    KolmogorovFlowNormalized,
)

from fluidsim.util.testing import classproperty
from fluidsim.solvers.ns3d.test_solver import TestSimulBase


class TestKolmo(TestSimulBase):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns3d.strat.solver import Simul as SimulNotExtended

        return extend_simul_class(
            SimulNotExtended, [KolmogorovFlow, KolmogorovFlowNormalized]
        )

    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "kolmogorov_flow"
        params.forcing.kolmo.ik = 3
        params.forcing.kolmo.amplitude = 2.0

        return params

    def test_kolmo(self):
        sim = self.sim
        sim.time_stepping.start()


class TestKolmoNormalized(TestKolmo):
    nx = 24

    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.type = "kolmogorov_flow_normalized"
        params.forcing.nkmax_forcing = 6
