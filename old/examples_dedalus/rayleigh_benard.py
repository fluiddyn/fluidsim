import numpy as np

from fluidsim.base.dedalus.solver import Simul

params = Simul.create_default_params()

params.prandtl = 1.0
params.rayleigh = 1e6
params.F = 1.0

params.output.sub_directory = "examples"
params.short_name_type_run = "rb"

params.oper.nx = 128
params.oper.nz = 64
params.oper.Lx = 2.0
params.oper.Lz = 1.0

params.time_stepping.t_end = 10
params.time_stepping.USE_CFL = True
params.time_stepping.deltat0 = 0.125

params.init_fields.type = "in_script"

params.output.periods_print.print_stdout = 0.1
params.output.periods_save.phys_fields = 0.5

# Intantiate the simulation object
sim = Simul(params)

# Initialize the fields
solver = sim.dedalus_solver
domain = sim.oper.domain

z = domain.grid(1)
b = solver.state["b"]
bz = solver.state["dz_b"]

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
z_basis = domain.bases[1]
zb, zt = z_basis.interval
pert = 1e-3 * noise * (zt - z) * (z - zb)
b["g"] = params.F * pert
b.differentiate("z", out=bz)

# In this case (params.init_fields.type = 'in_script') if we want to plot the
# result of the initialization before the time_stepping, we need to manually
# initialized the output:
#
# sim.output.init_with_initialized_state()
# sim.output.phys_fields.plot(key_field='b')

sim.time_stepping.start()
