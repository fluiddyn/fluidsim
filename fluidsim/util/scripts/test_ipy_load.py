import sys
from unittest.mock import patch

from fluidsim.util.scripts.ipy_load import start_ipython_load_sim


def test_start_ipython_load_sim(mocker):
    mocker.patch("IPython.start_ipython")
    with patch.object(sys, "argv", ["fluidsim-ipy-load"]):
        start_ipython_load_sim()
