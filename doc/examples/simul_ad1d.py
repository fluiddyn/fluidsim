from fluidsim.solvers.ad1d.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"

params.U = 1.0

params.oper.nx = 200
params.oper.Lx = 1.0

params.time_stepping.type_time_scheme = "RK2"

params.nu_2 = 0.01

params.time_stepping.t_end = 0.4
params.time_stepping.USE_CFL = True

params.init_fields.type = "gaussian"

params.output.periods_print.print_stdout = 0.25

params.output.periods_save.phys_fields = 0.1

params.output.periods_plot.phys_fields = 0.0

params.output.phys_fields.field_to_plot = "s"

sim = Simul(params)

sim.time_stepping.start()
