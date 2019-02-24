"""Script for a short simulation with the solver ns2d.strat

"""
import numpy as np
from math import pi

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()

gamma = 1.0
nb_wavelength = 2
nz = nb_wavelength * 30 * 2
nx = 6 * nz

params.time_stepping.t_end = 100.0

params.short_name_type_run = "quasi1d"

aspect_ratio = nz / nx
theta = np.arctan(aspect_ratio / nb_wavelength)
F = np.sin(theta)

kf = 2 * pi
omega_af = 2 * pi
forcing_rate = omega_af ** 3 / kf ** 2

Lx = 2 * pi / kf * np.sqrt(1 + (nb_wavelength / aspect_ratio) ** 2)
Lz = aspect_ratio * Lx

params.oper.nx = nx
params.oper.ny = nz
params.oper.Lx = Lx
params.oper.Ly = Lz
params.oper.coef_dealiasing = 0.65

# Brunt Vaisala frequency
params.N = gamma * omega_af / F

k_max = params.oper.coef_dealiasing * pi * nx / Lx
omega_max = (forcing_rate * k_max ** 2) ** (1 / 3)
params.nu_8 = 8e0 * omega_max / k_max ** 8

Uf = (forcing_rate / kf) ** (1 / 3)
params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 0.05 * Uf

params.NO_SHEAR_MODES = True
params.oper.NO_SHEAR_MODES = True

# forcing params
# key_forced = "rot_fft"
key_forced = "ap_fft"
neg_enable = True
# neg_enable = False

params.forcing.enable = True
params.forcing.type = "tcrandom_anisotropic"
params.forcing.tcrandom_anisotropic.angle = theta
params.forcing.nkmin_forcing = 0.01
params.forcing.nkmax_forcing = 2.1
params.forcing.forcing_rate = forcing_rate
params.forcing.key_forced = key_forced
params.forcing.tcrandom_anisotropic.kz_negative_enable = neg_enable
params.forcing.normalized.constant_rate_of = "energy"

params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.spectra_multidim = 1.0
params.output.periods_save.increments = 1.0

sim = Simul(params)

sim.time_stepping.start()

mpi.printby0(
    "\nTo display a video of this simulation, you can do:\n"
    f"cd {sim.output.path_run}"
    + """
ipython

# then in ipython (copy the 3 lines in the terminal):

from fluidsim import load_sim_for_plot
sim = load_sim_for_plot()

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.1, dt_equations=0.1)
"""
)
