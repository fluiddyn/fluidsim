import pytest

import numpy as np

import h5py

from fluidsim.util.phys_fields import (
    name_file_from_time_approx,
    compute_file_name,
    time_from_path,
)

from fluiddyn.util import mpi


dt = 0.0004
times = dt * np.arange(11)


@pytest.fixture(scope="module")
def path_dir_with_files(tmp_path_factory):
    path_dir = tmp_path_factory.mktemp("run_dir")

    str_width, ext = 7, "h5"

    for it, time in enumerate(times):
        path_file = path_dir / compute_file_name(time, str_width, ext)

        if path_file.exists():
            path_file = path_dir / compute_file_name(time, str_width, ext, it)

        with h5py.File(path_file, "w") as file:
            file.attrs["time"] = time
            file.attrs["it"] = it

    return path_dir


def test_time_from_path(path_dir_with_files):
    paths = sorted(path_dir_with_files.glob("*.h5"))
    for it, path in enumerate(paths):
        t_from_path = time_from_path(path)
        assert t_from_path == round(it * dt, 3)
        t_exact_from_path = time_from_path(path, exact=True)
        assert t_exact_from_path == it * dt


def test_name_file_from_time_approx(path_dir_with_files):
    if mpi.rank > 0:
        return

    path_dir = path_dir_with_files

    name_file_last = name_file_from_time_approx(path_dir)
    assert name_file_last == "state_phys_t000.004_it10.h5"

    name_file = name_file_from_time_approx(path_dir, t_approx=0.0034)
    assert name_file == "state_phys_t000.003_it8.h5", name_file

    name_file_last = name_file_from_time_approx(path_dir, t_approx="last")
    assert name_file_last == "state_phys_t000.004_it10.h5"
