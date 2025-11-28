import numpy as np

import h5py

from fluidsim.util.phys_fields import (
    name_file_from_time_approx,
    compute_file_name,
)


def test_name_file_from_time_approx(tmp_path):
    dt = 0.0004
    times = dt * np.arange(11)

    path_dir = tmp_path / "run_dir"
    path_dir.mkdir()

    str_width, ext = 7, "h5"

    for it, time in enumerate(times):
        path_file = path_dir / compute_file_name(time, str_width, ext)

        if path_file.exists():
            path_file = path_dir / compute_file_name(time, str_width, ext, it)

        with h5py.File(path_file, "w") as file:
            file.attrs["time"] = time
            file.attrs["it"] = it

        # print(path_file)

    name_file_last = name_file_from_time_approx(path_dir)
    assert name_file_last == path_file.name

    # TODO: add assert statements to get bugs!
