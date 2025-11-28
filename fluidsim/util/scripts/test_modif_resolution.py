import unittest

from pathlib import Path

import h5py

from fluiddyn.util import mpi
from fluiddyn.util.paramcontainer import ParamContainer

from fluidsim.util.testing import skip_if_no_fluidfft
from fluidsim.util.scripts.test_restart import path_simul
from fluidsim.util.scripts.modif_resolution import main


@unittest.skipIf(mpi.nb_proc > 1, "Modif resolution do not work with mpi")
@skip_if_no_fluidfft
def test_with_t_approx(path_simul):
    args = [path_simul, "2"]
    main(args, t_approx=1000)
    assert list(Path(path_simul).glob("State_phys_24x24/state_phys_t*"))


@unittest.skipIf(mpi.nb_proc > 1, "Modif resolution do not work with mpi")
@skip_if_no_fluidfft
def test_with_path_file(path_simul):
    path_simul = Path(path_simul)
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    args = [path_last_time, "3/2"]
    main(args)

    paths = list(path_simul.glob("State_phys_18x18/state_phys_t*"))
    assert paths

    with h5py.File(paths[0], "r") as file:
        state_params = ParamContainer(hdf5_object=file["/state_params"])
    assert state_params.a_tag.coef0 == 1.0
