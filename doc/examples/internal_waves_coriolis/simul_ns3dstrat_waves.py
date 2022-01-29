"""Simulation similar to experiments in Coriolis with two plates oscillating to
force internal waves in the Coriolis platform.

Launch with::

  mpirun -np 4 python simul_ns3dstrat_waves.py

"""

from math import pi

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.strat.solver import Simul

# main input parameters
N = 0.6  # rad/s
# if aspect_ratio == 4:
omega_f = 0.807 * N  # rad/s
# elif aspect_ratio == 6:
#    omega_f = 0.674 * N  # rad/s

delta_omega_f = 0.1 * omega_f  # rad/s

amplitude = 0.05

# useful parameters and secondary input parameters
period_N = 2 * pi / N
# total period of the forcing signal
period_forcing = 1e3 * period_N

params = Simul.create_default_params()

params.N = N

params.output.sub_directory = "waves_coriolis"

nz = 30
aspect_ratio = 4
nx = nz * aspect_ratio
ny = nx
lz = 1

params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = lx = lz / nz * nx
params.oper.Ly = ly = lz / nz * ny
params.oper.Lz = lz
params.oper.NO_SHEAR_MODES = True
params.no_vz_kz0 = True

r"""

Order of magnitude of nu_8?
---------------------------

Since the dissipation frequency is $\nu_n k^n$, we can define a Reynolds number
as:

$$Re_n = \frac{U L^{n-1}}{\nu_n}.$$

If we take a turbulent scaling $u(l) = (\varepsilon l)^{1/3}$, we obtain

$$Re_n(l) = \frac{\varepsilon^{1/3} l^(n - 2/3)}{\nu_n}.$$

The Kolmogorov length scale $\eta_n$ can be defined as the scale for which
$Re_n(l) = 1$:

$$ {\eta_n}^{n - 2/3} = \frac{\varepsilon^{1/3}}{\nu_n} $$

We want that $dx < \eta_n$, so we choose $\nu_n$ such that $dx = C \eta_n$
where $C$ is a constant of order 1.

"""
n = 8
C = 1.0
dx = lx / nx
U = amplitude * omega_f
H = 1
eps = 1e-2 * U**3 / H
params.nu_8 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 20 * period_N
params.time_stepping.deltat_max = deltat_max = period_N / 40

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 0.001
params.init_fields.noise.length = 2e-1

params.forcing.enable = True
params.forcing.type = "watu_coriolis"

watu = params.forcing.watu_coriolis
watu.omega_f = omega_f
watu.delta_omega_f = delta_omega_f
watu.amplitude = amplitude
watu.period_forcing = period_forcing
watu.approximate_dt = period_N / 1e1
watu.nb_wave_makers = 2

params.output.periods_print.print_stdout = 4.0

params.output.periods_save.phys_fields = 16.0
params.output.periods_save.spectra = 4.0
params.output.periods_save.spatial_means = 4.0
params.output.periods_save.spect_energy_budg = 1.0

params.output.spectra.kzkh_periodicity = 2

sim = Simul(params)

sim.time_stepping.start()

mpi.printby0(
    f"""
# To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

# To visualize with fluidsim:

cd {sim.output.path_run}
ipython

# in ipython:

from fluidsim import load_sim_for_plot
sim = load_sim_for_plot()
sim.output.phys_fields.set_equation_crosssection('x={lx/2}')
sim.output.phys_fields.animate('b')

"""
)

if mpi.rank == 0:
    sim.output.spect_energy_budg.plot_fluxes(key_k="kh")


import matplotlib.pyplot as plt

plt.show()
