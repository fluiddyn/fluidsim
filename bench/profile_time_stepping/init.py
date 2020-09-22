from fluidsim.solvers.ns2d.solver import Simul
from fluidsim.solvers.sw1l.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "bench_time_stepping"

params.oper.nx = params.oper.ny = nh = 1024

params.nu_8 = 1.0

params.time_stepping.it_end = 10
params.time_stepping.USE_T_END = False
params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = 1e-16
params.time_stepping.type_time_scheme = "RK2"

params.init_fields.type = "dipole"

params.forcing.enable = True
params.forcing.type = "proportional"

params.output.periods_print.print_stdout = 0.25

sim = Simul(params)

print("used time stepping func:\n", sim.time_stepping._time_step_RK)

"""
# cython
%timeit sim.time_stepping._time_step_RK()
# numpy
%timeit sim.time_stepping._time_step_RK2()
# pythran
%timeit sim.time_stepping._time_step_RK2_pythran()
# transonic
%timeit sim.time_stepping._time_step_RK2_transonic()

108 ms ± 292 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
94.1 ms ± 449 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
94.1 ms ± 441 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# cython
%timeit sim.time_stepping._time_step_RK()
# numpy
%timeit sim.time_stepping._time_step_RK4()
# transonic
%timeit sim.time_stepping._time_step_RK4_transonic()

243 ms ± 6.93 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
263 ms ± 5.38 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
227 ms ± 9.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""
