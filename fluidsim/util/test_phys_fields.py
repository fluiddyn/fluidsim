import numpy as np

from fluidsim.util.phys_fields import name_file_from_time_approx


def test_name_file_from_time_approx():
    dt = 0.0002
    times = dt * np.arange(10)
