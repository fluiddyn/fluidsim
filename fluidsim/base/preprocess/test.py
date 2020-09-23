import unittest

import numpy as np

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral

from fluidsim.util.testing import TestSimul, skip_if_no_fluidfft


@skip_if_no_fluidfft
class TestPreprocessPS(TestSimul):

    Simul = SimulBasePseudoSpectral

    @classmethod
    def init_params(cls):

        cls.params = params = SimulBasePseudoSpectral.create_default_params()
        params.short_name_type_run = "test_preprocess_ps"
        nh = 16
        Lh = 2 * np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh
        params.init_fields.type = "constant"
        params.preprocess.enable = True
        params.preprocess.viscosity_type = "laplacian"
        params.preprocess.viscosity_scale = "energy"

    def test_set_viscosity(self):
        """Tests if preprocess can initialize viscosity or not."""
        sim = self.sim
        self.assertGreater(sim.params.nu_2, 0)
        self.assertEqual(sim.params.nu_4 + sim.params.nu_8, 0)
        assert np.any(sim.time_stepping.freq_lin > 0)
        sim.output.close_files()


if __name__ == "__main__":
    unittest.main()
