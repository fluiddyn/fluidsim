from fluidsim_core.loader import import_cls_simul
from fluidsim_core.tests.solver import SimulTest


def test_load_cls_simul():
    Simul = import_cls_simul("test", "fluidsim_core.tests")
    assert Simul is SimulTest
