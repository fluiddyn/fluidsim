from fluidsim.base.solvers.base import InfoSolverBase


class InfoSolverFiniteDiff(InfoSolverBase):
    """Contain the information on a solver."""

    def _init_root(self):

        super()._init_root()

        # self.classes.State.module_name = 'fluidsim.base.state'
        # self.classes.State.class_name = 'StateBase'

        self.classes.TimeStepping.module_name = (
            "fluidsim.base.time_stepping.finite_diff"
        )
        self.classes.TimeStepping.class_name = (
            "TimeSteppingFiniteDiffCrankNicolson"
        )

        self.classes.Operators.module_name = "fluidsim.operators.op_finitediff1d"
        self.classes.Operators.class_name = "OperatorFiniteDiff1DPeriodic"


# # TODO: When required write a PreprocessFiniteDiff class.
# self.classes._set_child(
#     'Preprocess',
#     attribs={'module_name':
#              'fluidsim.base.preprocess.base',
#              'class_name': 'PreprocessBase'})
