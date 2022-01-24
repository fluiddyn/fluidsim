import sys
import shutil

from unittest.mock import MagicMock, patch

from fluidsim.util.scripts import turb_trandom_anisotropic
from fluidsim.util.scripts.turb_trandom_anisotropic import main

from fluiddyn.util import mpi


@patch.object(turb_trandom_anisotropic.plt, "show", MagicMock)
@patch.object(sys, "argv", ["./prog", "-opf"])
def test_only_plot_forcing():
    main(nz=12, ratio_nh_nz=2)


@patch.object(sys, "argv", ["./prog", "-opp", "--projection", "polo"])
def test_only_print_params():
    main()


@patch.object(
    sys, "argv", ["./prog", "-oppac", "--nu4", "1e-4", "--spatiotemporal-spectra"]
)
def test_only_print_params_as_code():
    main()


@patch.object(sys, "argv", ["./prog"])
def test_simul():
    params, sim = main(t_end=0.1, nz=12, ratio_nh_nz=2)
    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run, ignore_errors=True)
