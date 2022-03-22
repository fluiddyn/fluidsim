from pathlib import Path
import unittest

import numpy as np
import h5py

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.test_solver import (
    TestSimulBase,
    skip_if_no_fluidfft,
    classproperty,
)

from fluidsim.util import (
    modif_resolution_from_dir,
    modif_resolution_from_dir_memory_efficient,
)


@unittest.skipIf(mpi.nb_proc > 1, "Modif resolution do not work with mpi")
@skip_if_no_fluidfft
class TestModifResol3d(TestSimulBase):
    def test_modif(self):
        self.sim.output.phys_fields.save()
        self.sim.output.close_files()

        path_run = self.sim.output.path_run

        # input parameters of the modif_resol functions
        t_approx = None
        coef_modif_resol = 3 / 2

        # first, the standard function
        modif_resolution_from_dir(
            path_run, t_approx, coef_modif_resol, PLOT=False
        )
        path_big = next(Path(path_run).glob("State_phys_*/state_phys*"))
        path_big_old = path_big.with_name("old_" + path_big.name)
        path_big.rename(path_big_old)

        # Then, the alternative implementation
        modif_resolution_from_dir_memory_efficient(
            path_run, t_approx, coef_modif_resol
        )

        with h5py.File(path_big_old, "r") as file:
            group_state_phys = file["/state_phys"]
            key = list(group_state_phys.keys())[0]
            field_old = group_state_phys[key][...]

        with h5py.File(path_big, "r") as file:
            group_state_phys = file["/state_phys"]
            field = group_state_phys[key][...]

        assert np.allclose(field_old, field)


class TestModifResol2d(TestModifResol3d):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns2d.solver import Simul

        return Simul
