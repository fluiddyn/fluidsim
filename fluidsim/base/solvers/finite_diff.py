
from fluidsim.base.solvers.base import InfoSolverBase


class InfoSolverFiniteDiff(InfoSolverBase):
    """Contain the information on a solver."""

    def _init_root(self):

        super(InfoSolverFiniteDiff, self)._init_root()

        self.classes._set_child(
            'State',
            attribs={'module_name': 'fluidsim.base.state',
                     'class_name': 'StateBase'})

        self.classes._set_child(
            'TimeStepping',
            attribs={'module_name':
                     'fluidsim.base.time_stepping.finite_diff',
                     'class_name':
                     'TimeSteppingFiniteDiffCrankNicolson'})

        self.classes._set_child(
            'Operators',
            attribs={'module_name':
                     'fluidsim.operators.op_finitediff',
                     'class_name': 'OperatorFiniteDiff1DPeriodic'})
        
        self.classes._set_child(
            'Preprocess',
            attribs={'module_name':
                     'fluidsim.base.preprocess.base',
                     'class_name': 'PreprocessBase'}) # TODO: When required write a PreprocessFiniteDiff class.
