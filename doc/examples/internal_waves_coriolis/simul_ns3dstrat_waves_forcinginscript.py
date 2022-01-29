"""Simulation similar to experiments in Coriolis with two plates oscillating to
force internal waves in the Coriolis platform.

Launch with::

  mpirun -np 4 python simul_ns3dstrat_waves.py

"""

from math import pi
import numpy as np

from scipy.interpolate import interp1d

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.strat.solver import Simul

from fluidsim.util.frequency_modulation import FrequencyModulatedSignalMaker

# main input parameters
omega_f = 0.3  # rad/s
delta_omega_f = 0.03  # rad/s
N = 0.4  # rad/s
amplitude = 0.05  # m

# useful parameters and secondary input parameters
period_N = 2 * pi / N
# total period of the forcing signal
period_forcing = 1e3 * period_N

# preparation of a time signal for the forcing
if mpi.rank == 0:
    time_signal_maker = FrequencyModulatedSignalMaker(
        total_time=period_forcing, approximate_dt=period_N / 1e1
    )

    def create_interpolation_forcing_function():
        (
            times_forcing,
            forcing_vs_time,
        ) = time_signal_maker.create_frequency_modulated_signal(
            omega_f, delta_omega_f, amplitude
        )

        return interp1d(times_forcing, forcing_vs_time, fill_value="extrapolate")

    calcul_forcing_time_x = create_interpolation_forcing_function()
    calcul_forcing_time_y = create_interpolation_forcing_function()
else:
    calcul_forcing_time_x = calcul_forcing_time_y = None

if mpi.nb_proc > 1:
    calcul_forcing_time_x = mpi.comm.bcast(calcul_forcing_time_x, root=0)
    calcul_forcing_time_y = mpi.comm.bcast(calcul_forcing_time_y, root=0)


# initialization of the simulation

params = Simul.create_default_params()

params.N = N

params.output.sub_directory = "waves_coriolis"

nz = 64
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
n = 2
C = 1.0
dx = lx / nx
U = amplitude * omega_f
H = 1
eps = 1e-2 * U**3 / H
params.nu_2 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 50 * period_N
params.time_stepping.deltat_max = deltat_max = period_N / 40

params.init_fields.type = "noise"
params.init_fields.noise.velo_max = 0.001
params.init_fields.noise.length = 2e-1

params.forcing.enable = True
params.forcing.type = "in_script"

params.output.periods_print.print_stdout = 1.0

params.output.periods_save.phys_fields = 2.0
params.output.periods_save.spectra = 1.0
params.output.periods_save.spatial_means = 0.5

sim = Simul(params)

# monkey-patching for forcing
oper = sim.oper
X, Y, Z = oper.get_XYZ_loc()


# calculus of the target velocity components

width = max(4 * dx, amplitude / 5)


def step_func(x):
    """Activation function"""
    return 0.5 * (np.tanh(x / width) + 1)


amplitude_side = amplitude + 0.15

maskx = (
    (step_func(-(X - amplitude)) + step_func(X - (lx - amplitude)))
    * step_func(Y - amplitude_side)
    * step_func(-(Y - (ly - amplitude_side)))
)

masky = (
    (step_func(-(Y - amplitude)) + step_func(Y - (ly - amplitude)))
    * step_func(X - amplitude_side)
    * step_func(-(X - (lx - amplitude_side)))
)

z_variation = np.sin(2 * pi * Z)
vxtarget = z_variation
vytarget = z_variation


# calculus of coef_sigma
"""
f(t) = f0 * exp(-sigma*t)

If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:

sigma = gamma / (n_dt * dt)
"""
gamma = 2
n_dt = 4
coef_sigma = gamma / n_dt


def compute_forcing_each_time(self):
    """This function is called by the forcing_maker to compute the forcing"""
    sim = self.sim
    time = sim.time_stepping.t % period_forcing
    coef_forcing_time_x = calcul_forcing_time_x(time)
    coef_forcing_time_y = calcul_forcing_time_y(time)
    vx = sim.state.state_phys.get_var("vx")
    vy = sim.state.state_phys.get_var("vy")
    sigma = coef_sigma / deltat_max
    fx = sigma * maskx * (coef_forcing_time_x * vxtarget - vx)
    fy = sigma * masky * (coef_forcing_time_y * vytarget - vy)
    result = {"vx_fft": fx, "vy_fft": fy}
    return result


sim.forcing.forcing_maker.monkeypatch_compute_forcing_each_time(
    compute_forcing_each_time
)

# finally we start the simulation
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
