"""Script for a short simulation with the solver ns3d.bouss

The field initialization is done in the script.

Launch with::

  mpirun -np 4 python simul_ns3dstrat_check_energy_conserv.py

"""

from fluidsim.solvers.ns3d.strat.solver import Simul

params = Simul.create_default_params()

params.f = 1.0

params.output.sub_directory = "examples"

nx = 48
ny = 64
nz = 96
Lx = 3
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny
params.oper.Lz = Lz = Lx / nx * nz

params.short_name_type_run = "checkenergy"

params.oper.coef_dealiasing = 0.5

params.time_stepping.USE_T_END = False
params.time_stepping.USE_CFL = False
params.time_stepping.it_end = 10
params.time_stepping.deltat0 = 1e-2

params.init_fields.type = "noise"

params.output.periods_print.print_stdout = 1e-10
params.output.periods_save.phys_fields = 0.0

sim = Simul(params)

sim.time_stepping.start()
