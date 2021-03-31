from fluidsim_core.tests.solver import SimulTest


def test_init_sim():
    params = SimulTest.create_default_params()
    assert params.foo == 42
    params.foo = 24
    sim = SimulTest(params)
    assert sim.params.foo == 24
