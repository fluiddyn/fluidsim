

import unittest
import shutil
from glob import glob
import os


import numpy as np

import matplotlib

matplotlib.use("Agg")

import fluiddyn as fld
from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi

# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral
from fluidsim import modif_resolution_from_dir, load_params_simul

from fluidsim.base.params import load_info_solver


class TestBaseSolverPS(unittest.TestCase):
    def setUp(self, params=None):
        """Should be able to run a TestBaseSolverPS simulation."""

        self.cwd = os.getcwd()

        if params is None:
            params = SimulBasePseudoSpectral.create_default_params()
            params.output.periods_plot.phys_fields = 0.
            params.output.periods_print.print_stdout = 0.
            params.short_name_type_run = "test_base_solver_ps"

        nh = 8
        Lh = 2 * np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.nu_2 = 1.

        params.time_stepping.t_end = 0.4

        with stdout_redirected():
            self.sim = SimulBasePseudoSpectral(params)

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, "sim"):
                self.sim.output.print_stdout.close()
                shutil.rmtree(self.sim.output.path_run)

            os.chdir(self.cwd)

    def test_simul(self):
        """Should be able to run a base experiment."""
        with stdout_redirected():
            self.sim.time_stepping.start()
            load_params_simul(
                self.sim.output.path_run + "/params_simul.xml",
                only_mpi_rank0=False
            )

        fld.show()

        if mpi.nb_proc > 1:
            return

        with stdout_redirected():
            modif_resolution_from_dir(
                self.sim.output.path_run, coef_modif_resol=3./2, PLOT=False
            )
            path_new = os.path.join(self.sim.output.path_run, "State_phys_12x12")
            os.chdir(path_new)
            load_params_simul()
            path = glob("state_*")[0]
            load_params_simul(path)
            load_info_solver()


class TestOutputPS(TestBaseSolverPS):
    """Test a simulation run with online plotting and stdout printing."""

    def setUp(self):
        params = SimulBasePseudoSpectral.create_default_params()
        params.output.periods_plot.phys_fields = 0.2
        params.output.periods_print.print_stdout = 0.2
        params.short_name_type_run = "test_output_ps"
        TestBaseSolverPS.setUp(self, params)


if __name__ == "__main__":
    unittest.main()
