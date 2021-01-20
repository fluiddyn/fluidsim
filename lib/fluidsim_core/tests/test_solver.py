from fluidsim_core.info import InfoSolverCore
from fluidsim_core.solver import SimulCore


class InfoSolverTest(InfoSolverCore):
    """Contain the information on a solver."""

    def _init_root(self):
        super()._init_root()

        self._set_attribs(
            {
                "module_name": "fluidsim_core.tests.test_solver",
                "class_name": "SimulTest",
                "short_name": "Test",
            }
        )


class SimulTest(SimulCore):
    InfoSolver = InfoSolverTest

    @staticmethod
    def _complete_params_with_default(params):
        params._set_attribs({"foo": 42, "bar": True})

    @classmethod
    def create_default_params(cls):
        return super().create_default_params()

    def __init__(self, params):
        super().__init__(params)


# All unit tests


def test_init_sim():
    params = SimulTest.create_default_params()
    SimulTest(params)
