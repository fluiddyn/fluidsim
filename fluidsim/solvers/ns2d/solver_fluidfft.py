
from .solver import InfoSolverNS2D as _InfoSolverNS2D, Simul as _Simul

class InfoSolverNS2D(_InfoSolverNS2D):
    def _init_root(self):
        super(InfoSolverNS2D, self)._init_root()
        self.classes.Operators.module_name = 'fluidsim.operators.operators2d'
        self.classes.Operators.class_name = 'OperatorsPseudoSpectral2D'

        
class Simul(_Simul):
    InfoSolver = InfoSolverNS2D


if __name__ == "__main__":
    import numpy as np
    
    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 16
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 5.

    params.init_fields.type = 'noise'

    params.output.periods_plot.phys_fields = 0.

    params.output.periods_print.print_stdout = 0.25
    params.output.periods_save.phys_fields = 2.

    sim = Simul(params)

    sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
