import unittest

import numpy as np

try:
    from fluidsht.sht2d.operators import get_simple_2d_method

    get_simple_2d_method()
    sht_avail = True
except (ImportError, NotImplementedError):
    sht_avail = False


from fluidsim.solvers.sphere.ns2d.solver import Simul
from fluidsim.util.testing import TestSimul


class TestSimulBase(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()
        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.oper.lmax = 15
        params.oper.omega = 0.0

        params.nu_8 = 2.0

        params.time_stepping.t_end = 0.5

        params.init_fields.type = "noise"

        return params


@unittest.skipUnless(sht_avail, "No SHT transform library available")
class TestSolverSphereNS2DTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.output.HAS_TO_SAVE = False

    @unittest.expectedFailure
    def test_tendency(self):
        sim = self.sim
        rot_sh = sim.state.get_var("rot_sh")

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Frot_sh = tend.get_var("rot_sh")

        T_rot = np.real(Frot_sh.conj() * rot_sh)

        ratio = sim.oper.sum_wavenumbers(T_rot) / sim.oper.sum_wavenumbers(
            abs(T_rot)
        )

        self.assertGreater(1e-15, abs(ratio))
