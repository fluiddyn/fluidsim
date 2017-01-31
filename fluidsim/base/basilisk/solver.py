

from fluidsim.base.solvers.base import SimulBase
from fluidsim.base.solvers.info_base import InfoSolverBase


class InfoSolverBasilisk(InfoSolverBase):

    def _init_root(self):

        super(InfoSolverBasilisk, self)._init_root()

        mod = 'fluidsim.base.basilisk'

        self.module_name = mod + '.solver'
        self.class_name = 'SimulBasilisk'
        self.short_name = 'bas'

        classes = self.classes

        classes.State.module_name = mod + '.state'
        classes.State.class_name = 'StateBasilisk'

        classes.TimeStepping.module_name = mod + '.time_stepping'
        classes.TimeStepping.class_name = 'TimeSteppingBasilisk'



class SimulBasilisk(SimulBase):
    InfoSolver = InfoSolverBasilisk


Simul = SimulBasilisk

if __name__ == "__main__":

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    sim = Simul(params)
    # sim.time_stepping.start()
