import sys
import shutil
import unittest
from unittest.mock import MagicMock, patch

from fluiddyn.util import mpi

from fluidsim.util.testing import skip_if_no_fluidfft
from fluidsim.util.scripts import turb_trandom_anisotropic
from fluidsim.util.scripts.turb_trandom_anisotropic import main


@unittest.skipIf(mpi.nb_proc > 1, "Modif resolution do not work with mpi")
@patch.object(turb_trandom_anisotropic.plt, "show", MagicMock)
@patch.object(sys, "argv", ["./prog", "-opf"])
@skip_if_no_fluidfft
def test_only_plot_forcing():
    main(nz=12, ratio_nh_nz=2)


@patch.object(sys, "argv", ["./prog", "-opp", "--projection", "polo"])
@skip_if_no_fluidfft
def test_only_print_params():
    main()


@patch.object(
    sys, "argv", ["./prog", "-oppac", "--nu4", "1e-4", "--spatiotemporal-spectra"]
)
@skip_if_no_fluidfft
def test_only_print_params_as_code():
    main()


@patch.object(
    sys, "argv", ["./prog", "-oppac", "--Rb4", "10", "--spatiotemporal-spectra"]
)
@skip_if_no_fluidfft
def test_only_print_params_as_code_Rb4():
    main()


@patch.object(sys, "argv", ["./prog"])
@skip_if_no_fluidfft
def test_simul():
    params, sim = main(t_end=0.1, nz=12, ratio_nh_nz=2)
    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run, ignore_errors=True)
