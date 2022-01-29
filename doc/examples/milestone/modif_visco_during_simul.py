"""

"""

from math import pi

import matplotlib.pyplot as plt

from fluidsim.solvers.ns3d.solver import Simul
from fluidsim.base.forcing.milestone import PeriodicUniform

sub_directory = "milestone-modif-visco"

diameter = 0.5
speed = 0.1
number_cylinders = 1
ny_per_cylinder = 16


params = Simul.create_default_params()

mesh = 3 * diameter

ly = params.oper.Ly = mesh * number_cylinders
lx = params.oper.Lx = mesh * 4
ny = params.oper.ny = ny_per_cylinder * number_cylinders

nx_float = ny * lx / ly
nx = params.oper.nx = round(nx_float)
assert nx == nx_float, nx_float

dx = lx / nx

lz = params.oper.Lz = mesh
params.oper.nz = round(lz / dx)

params.oper.NO_SHEAR_MODES = True

params.forcing.enable = True
params.forcing.type = "milestone"
params.forcing.milestone.nx_max = 64
objects = params.forcing.milestone.objects

objects.number = number_cylinders
objects.diameter = diameter
objects.width_boundary_layers = 0.2 * diameter

movement = params.forcing.milestone.movement

movement.type = "periodic_uniform"
movement.periodic_uniform.length = lx - 2 * diameter
movement.periodic_uniform.length_acc = diameter / 2
movement.periodic_uniform.speed = speed

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = speed
params.init_fields.noise.length = diameter

movement = PeriodicUniform(
    speed,
    movement.periodic_uniform.length,
    movement.periodic_uniform.length_acc,
    lx,
)

params.time_stepping.t_end = movement.period * 0.25
params.time_stepping.deltat_max = 0.1 * diameter / speed

epsilon_eval = 0.02 * speed**3 / mesh
kmax = params.oper.coef_dealiasing * pi / dx
eta_kmax = 2 * pi / kmax
nu_2_needed = (epsilon_eval * eta_kmax**4) ** (1 / 3)
freq_nu4 = 0.5 * (nu_2_needed - params.nu_2) * kmax**2

nu_4_needed = freq_nu4 / kmax**4

# for the first main time loop, nearly no viscosity!
# warning: not exactly 0 otherwise, output methods get lost
params.nu_4 = 1e-14

params.output.sub_directory = sub_directory
params.output.periods_print.print_stdout = movement.period / 20.0

periods_save = params.output.periods_save
periods_save.phys_fields = movement.period / 10.0
periods_save.spatial_means = movement.period / 1000.0
periods_save.spect_energy_budg = movement.period / 50.0
periods_save.spectra = movement.period / 100.0

sim = Simul(params)

"""
For this simulation, we want to

- run a first time loop,
- modify t_end and nu_4,
- relaunch the time loop.

The usual call of `sim.time_stepping.start` is replaced by these few lines:
"""

sim.time_stepping.main_loop(print_begin=True, save_init_field=True)

print("first main loop finished, let's start a new one")

# modify t_end and nu_4
assert sim.params is params
params.time_stepping.t_end *= 2
params.nu_4 = nu_4_needed

# time_stepping parameters have been changed, we need to call
sim.time_stepping.init_from_params()

sim.time_stepping.main_loop()

sim.time_stepping.finalize_main_loop()

# some plots to visualize the result
sim.output.spatial_means.plot()
sim.output.spectra.plot1d(coef_compensate=5 / 3)

plt.show()
