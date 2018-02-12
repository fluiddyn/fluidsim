
from math import pi

import fluiddyn as fld
from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()

params.short_name_type_run = 'test'

params.oper.nx = params.oper.ny = nh = 32
params.oper.Lx = params.oper.Ly = Lh = 2 * pi

delta_x = Lh / nh
params.nu_8 = 2.*params.forcing.forcing_rate**(1./3)*delta_x**8

params.time_stepping.t_end = 10.

params.init_fields.type = 'dipole'

params.FORCING = True
params.forcing.type = 'tcrandom'

params.output.sub_directory = 'examples'

params.output.periods_plot.phys_fields = 0.1

params.output.ONLINE_PLOT_OK = True


sim = Simul(params)
sim.time_stepping.start()
fld.show()
