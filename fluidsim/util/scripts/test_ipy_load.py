from fluidsim.util.scripts.ipy_load import start_ipython_load_sim


def test_start_ipython_load_sim(mocker):
    mocker.patch("IPython.start_ipython")
    start_ipython_load_sim()
