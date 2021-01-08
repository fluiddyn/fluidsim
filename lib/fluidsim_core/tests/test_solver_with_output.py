from fluidsim_core.info import InfoSolverCore
from fluidsim_core.output import OutputCore
from fluidsim_core.tests.test_solver import SimulTest


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
    def _complete_params_with_default(params):
        params._set_attribs({
            "NEW_DIR_RESULTS": True,
            "short_name_type_run": "test"
        })

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

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        OutputCore._complete_params_with_default(params, info_solver)

    def __init__(self, sim):
        super().__init__(sim)

    def post_init(self):
        super().post_init()


def test_init_sim_output():
    params = SimulTestWithOutput.create_default_params()
    SimulTestWithOutput(params)
