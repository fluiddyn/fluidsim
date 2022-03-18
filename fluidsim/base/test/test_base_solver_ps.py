import unittest
from glob import glob
import os

import numpy as np

import fluiddyn as fld
import fluiddyn.util.mpi as mpi

# to get fld.show
import fluiddyn.output

from fluidsim import (
    modif_resolution_from_dir,
    load_params_simul,
    load_state_phys_file,
)

from fluidsim.util import times_start_last_from_path

from fluidsim.base.params import load_info_solver

from fluidsim.util.testing import TestSimul, classproperty, skip_if_no_fluidfft


@skip_if_no_fluidfft
class TestBaseSolverPS(TestSimul):
    """Test of the base class for pseudo_spect solvers"""

    @classproperty
    def Simul(cls):
        from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral

        return SimulBasePseudoSpectral

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cwd = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.chdir(cls.cwd)

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()
        params.output.periods_plot.phys_fields = 0.2
        params.output.periods_print.print_stdout = 0.2
        params.short_name_type_run = "test_base_solver_ps"

        nh = 8
        Lh = 2 * np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.nu_2 = 1.0

        params.time_stepping.t_end = 0.4
        params.time_stepping.type_time_scheme = "RK2"

    def test_simul(self):
        """Should be able to run a base experiment."""
        self.sim.time_stepping.start()
        load_params_simul(
            self.sim.output.path_run + "/params_simul.xml", only_mpi_rank0=False
        )

        fld.show()

        if mpi.nb_proc > 1:
            return

        modif_resolution_from_dir(
            self.sim.output.path_run, coef_modif_resol=3.0 / 2, PLOT=True
        )

        times_start_last_from_path(self.sim.output.path_run)

        path_new = os.path.join(self.sim.output.path_run, "State_phys_12x12")
        os.chdir(path_new)
        load_params_simul()
        path = glob("state_*")[0]
        load_params_simul(path)
        load_info_solver()

        sim_big = load_state_phys_file(path_new)

        for key in self.sim.state.keys_state_phys:
            var = self.sim.state.get_var(key)
            var_big = sim_big.state.get_var(key)
            assert np.mean(var**2) == np.mean(var_big**2)


if __name__ == "__main__":
    unittest.main()
