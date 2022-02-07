from fluidsim_core.info import InfoSolverCore
from fluidsim_core.output import OutputCore
from fluidsim_core.params import Parameters, iter_complete_params
from fluidsim_core.tests.test_solver import SimulTest

import pytest


class InfoSolverTestWithOutput(InfoSolverCore):
    def _init_root(self):
        super()._init_root()

        self._set_attribs(
            {
                "module_name": "fluidsim_core.tests.test_solver_with_output",
                "class_name": "SimulTestWithOutput",
                "short_name": "TestWithOutput",
            }
        )
        classes = self.classes
        classes._set_child(
            "Output",
            attribs={
                "module_name": "fluidsim_core.tests.test_solver_with_output",
                "class_name": "OutputTest",
            },
        )


class SimulTestWithOutput(SimulTest):
    InfoSolver = InfoSolverTestWithOutput

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        params._set_attribs(
            {"NEW_DIR_RESULTS": True, "short_name_type_run": "test"}
        )

    def __init__(self, params):
        super().__init__(params)
        dict_classes = self.info_solver.import_classes()

        for cls_name, Class in dict_classes.items():
            # only initialize if Class is not the Simul class
            if not isinstance(self, Class):
                setattr(self, cls_name.lower(), Class(self))

        self.output.post_init()


class OutputTest(OutputCore):
    @staticmethod
    def _complete_info_solver(info_solver):
        OutputCore._complete_info_solver(info_solver)
        classes = info_solver.classes.Output.classes

        # NOTE: This is wrong and added to check if a corresponding UserWarning
        # is invoked
        classes._set_attrib(
            "Dummy",
            dict(
                module_name="fluidsim_core.tests.test_solver_with_output",
                class_name="DummyOutput",
            ),
        )

        # NOTE: This is correct :)
        classes._set_child(
            "Working",
            dict(
                module_name="fluidsim_core.tests.test_solver_with_output",
                class_name="WorkingOutput",
            ),
        )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        OutputCore._complete_params_with_default(params, info_solver)
        dict_classes = info_solver.classes.Output.import_classes()
        iter_complete_params(params, info_solver, dict_classes.values())

    def __init__(self, sim):
        super().__init__(sim)

        dict_classes = self.sim.info.solver.classes.Output.import_classes()
        # initialize objects
        for cls_name, Class in dict_classes.items():
            # only initialize if Class is not the Output class
            if not isinstance(self, Class):
                setattr(self, cls_name.lower(), Class(self))

    def post_init(self):
        super().post_init()


class SpecificOutputTest:
    def __init__(self, sim):
        pass


class DummyOutput(SpecificOutputTest):
    """This class is not specified in the InfoSolver instance
    properly (see ``_complete_info_solver`` method above) and thus won't
    be instantiated.

    """


class WorkingOutput(SpecificOutputTest):
    @staticmethod
    def _complete_params_with_default(params):
        params.output._set_child("working", attribs={"works": 42})


def assert_default_params(params):
    assert not hasattr(params.output, "dummy")
    assert hasattr(params.output, "working")
    assert params.output.working.works == 42


def test_init_info_solver_params():
    with pytest.warns(UserWarning, match=r"A class Dummy .* using _set_attrib"):
        info_solver = InfoSolverTestWithOutput()

    params = Parameters._create_params(info_solver)

    assert_default_params(params)


def test_init_sim_output():
    with pytest.warns(UserWarning, match=r"A class Dummy .* using _set_attrib"):
        params = SimulTestWithOutput.create_default_params()

    assert_default_params(params)
    params.output.sub_directory = "tests"
    sim = SimulTestWithOutput(params)

    assert hasattr(sim.output, "working")
