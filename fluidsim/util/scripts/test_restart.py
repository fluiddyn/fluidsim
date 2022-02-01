import shutil
import sys
from unittest.mock import patch
from pathlib import Path

import pytest

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.solver import Simul

from fluidsim.util.scripts.restart import main, restart


@pytest.fixture(scope="package")
def path_simul():
    params = Simul.create_default_params()
    params.nu_2 = 1e-3

    params.output.sub_directory = "tests"

    params.oper.nx = params.oper.ny = 12
    params.init_fields.type = "noise"
    params.init_fields.noise.length = 1.0
    params.init_fields.noise.velo_max = 1.0

    params.time_stepping.t_end = 1.5 * params.time_stepping.deltat_max
    params.time_stepping.it_end = 2

    sim = Simul(params)

    sim.time_stepping.start()

    yield sim.output.path_run

    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run, ignore_errors=True)


def test_only_check(path_simul):
    argv = ["fluidsim-restart", path_simul, "-oc", "--it_end", "1"]
    argv.extend(["--add-to-it_end", "1"])
    with patch.object(sys, "argv", argv):
        main()


def test_check_t_end(path_simul):
    path_simul = Path(path_simul)
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    argv = [path_last_time, "-oi"]
    restart(argv, add_to_it_end=2)


def test_only_init(path_simul):
    path_simul = Path(path_simul)
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    argv = ["fluidsim-restart", path_last_time, "-oi", "--t_end", "1.0"]
    with patch.object(sys, "argv", argv):
        main()


def test_check_it_end(path_simul):
    path_simul = Path(path_simul)
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    argv = [
        "fluidsim-restart",
        path_last_time,
        "--modify-params",
        "params.time_stepping.USE_T_END = False",
    ]
    with patch.object(sys, "argv", argv):
        main()


def test_simul(path_simul):
    argv = ["fluidsim-restart", path_simul, "--t_end", "1.0"]
    argv.extend(["--add-to-t_end", "1.0"])
    with patch.object(sys, "argv", argv):
        main()
