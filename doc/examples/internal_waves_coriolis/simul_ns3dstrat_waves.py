"""Simulation similar to experiments in Coriolis with two plates oscillating to
force internal waves in the Coriolis platform.

Launch with::

  mpirun -np 4 python simul_ns3dstrat_waves.py

"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from fluiddyn.util.mpi import printby0, rank
from fluiddyn.calcul.easypyfft import FFTW1DReal2Complex, fftw_grid_size

from fluidsim.solvers.ns3d.strat.solver import Simul

# main input parameters
omega_f_min = 0.2  # rad/s
omega_f_max = 0.4  # rad/s
N = 0.4  # rad/s
amplitude = 0.05  # m

# secondary input parameters

# total period of the forcing signal
period_forcing = 1e2*2*pi/N
dt_forcing = 2*pi/N/1e1


def step_func(x, width):
    return 0.5*(np.tanh(x/width) + 1)


# preparation of a time signal for the forcing
nt_forcing = 2 * fftw_grid_size(int(period_forcing/dt_forcing))
print(nt_forcing)
dt_forcing = period_forcing/nt_forcing
print('dt_forcing', dt_forcing)

oper_fft_forcing = FFTW1DReal2Complex(nt_forcing)
nomegas_forcing = oper_fft_forcing.shapeK[0]
forcing_omega = (np.random.rand(nomegas_forcing) +
                 1j*np.random.rand(nomegas_forcing))

domega = 2*pi/period_forcing
omegas_forcing = domega*np.arange(nomegas_forcing)
times_forcing = dt_forcing * np.arange(nt_forcing)

forcing_omega *= (step_func(omegas_forcing-omega_f_min, 4*domega) *
                  step_func(-omegas_forcing+omega_f_max, 4*domega))

# print(forcing_omega[omegas_forcing < 2*N])
forcing_time = oper_fft_forcing.ifft(forcing_omega)
forcing_time /= max(forcing_time)

calcul_forcing_time = interp1d(times_forcing, forcing_time,
                               fill_value='extrapolate')

# problem large values beginning and end...
# fig, ax = plt.subplots()
# ax.plot(times_forcing, forcing_time)
# plt.show()

# initialization of the simulation

params = Simul.create_default_params()

params.output.sub_directory = 'waves_coriolis'

nz = 48
nx = ny = nz*2
lz = 1

params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = lx = lz/nz*nx
params.oper.Ly = ly = lz/nz*ny
params.oper.Lz = lz

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
C = 1.
dx = lx/nx
U = amplitude*omega_f_max
H = 1
eps = 1e-2*U**3/H
params.nu_8 = (dx/C)**((3*n-2)/3) * eps**(1/3)

# printby0(f'nu_8 = {params.nu_8:.3e}')

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 8.
# we need small time step for a strong forcing
params.time_stepping.deltat_max = 0.01
params.time_stepping.cfl_coef = 0.5

params.init_fields.type = 'noise'
params.init_fields.noise.velo_max = 0.01

params.forcing.enable = True
params.forcing.type = 'in_script'

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5

sim = Simul(params)

# monkey-patching for forcing
oper = sim.oper
X, Y, Z = oper.get_XYZ_loc()

# calculus of the target velocity components
width = max(4*dx, 5e-3)
vxtarget = (step_func(-(X - amplitude), width) +
            step_func(X - (lx - amplitude), width))
vytarget = (step_func(-(Y - amplitude), width) +
            step_func(Y - (ly - amplitude), width))

z_variation = np.sin(2*pi*Z)
vxtarget *= z_variation
vytarget *= z_variation

# calculus of coef_sigma
"""
f(t) = f0 * exp(-sigma*t)

If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:

sigma = gamma / (n_dt * dt)
"""
gamma = 2
n_dt = 4
coef_sigma = gamma/n_dt


def compute_forcing_each_time(self):
    """This function is called by the forcing_maker to compute the forcing

    """
    coef_forcing_time = calcul_forcing_time(
        sim.time_stepping.t % period_forcing)
    vx = self.sim.state.state_phys.get_var('vx')
    vy = self.sim.state.state_phys.get_var('vy')
    sigma = coef_sigma/sim.time_stepping.deltat
    fx = sigma * (coef_forcing_time * vxtarget - vx)
    fy = sigma * (coef_forcing_time * vytarget - vy)
    result = {'vx_fft': fx, 'vy_fft': fy}
    return result

sim.forcing.forcing_maker.monkeypatch_compute_forcing_each_time(
    compute_forcing_each_time)

# finally we start the simulation
sim.time_stepping.start()

printby0(f"""
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

""")
