import matplotlib.pyplot as plt

from fluidsim.base.basilisk.solver import Simul

params = Simul.create_default_params()

params.short_name_type_run = "test"

params.oper.nx = 128

params.time_stepping.deltat0 = 2.4
params.output.periods_print.print_stdout = 1e-15

sim = Simul(params)
sim.time_stepping.start()

sim.output.print_stdout.plot_energy()
plt.show()
