
import shutil
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.solver import Simul

from fluidsim.util.scripts.restart import main

@pytest.fixture(scope="module")
def path_simul():
    params = Simul.create_default_params()
    params.nu_2 = 1e-3

    params.output.sub_directory = "tests"

    params.oper.nx = params.oper.ny = 12
    params.init_fields.type = "noise"
    params.init_fields.noise.length = 1.0
    params.init_fields.noise.velo_max = 1.0

    params.time_stepping.t_end = 2 * params.time_stepping.deltat_max

    sim = Simul(params)

    sim.time_stepping.start()

    yield sim.output.path_run

    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run, ignore_errors=True)


def test_only_check(path_simul):
    with patch.object(sys, "argv", ["fluidsim-restart", path_simul, "-oc"]):
        main()


def test_only_init(path_simul):
    path_simul = Path(path_simul)
    print(sorted(path_simul.glob("*")))
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    with patch.object(sys, "argv", ["fluidsim-restart", path_last_time, "-oi"]):
        main()


def test_simul(path_simul):
    with patch.object(sys, "argv", ["fluidsim-restart", path_simul, "--t_end", "1.0"]):
        main()
