"""Script for a short simulation with the solver ns3d.bouss

The field initialization is done in the script.

Launch with::

  mpirun -np 4 python simul_ns3dbouss_staticlayer.py

"""

import numpy as np

from fluiddyn.util.mpi import printby0

from fluidsim.solvers.ns3d.bouss.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"

nx = 144 // 2
ny = nx
nz = nx // 2
lz = 2
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = lx = lz / nz * nx
params.oper.Ly = ly = lz / nz * ny
params.oper.Lz = lz

# rotation
params.f = 1.0

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
B = 1
D = 1
eps = 1e-2 * B ** (3 / 2) * D ** (1 / 2)
params.nu_8 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

# printby0(f'nu_8 = {params.nu_8:.3e}')

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 4.0
params.time_stepping.deltat_max = 0.01

params.init_fields.type = "in_script"

params.forcing.enable = True
params.forcing.type = "in_script"

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.05

sim = Simul(params)

# here we have to initialize the flow fields
variables = {
    k: 1e-2 * sim.oper.create_arrayX_random() for k in ("vx", "vy", "vz")
}

X, Y, Z = sim.oper.get_XYZ_loc()

width_step = max(4 * dx, 0.1)


def step_func(x):
    """Activation function"""
    return 0.5 * (np.tanh(x / width_step) + 1)


x0 = lx / 2.0
y0 = ly / 2.0
Rh2 = (X - x0) ** 2 + (Y - y0) ** 2
r0 = 0.5
d_forcing = lz / 10
d_forcing_b = 1.2 * d_forcing

b = (
    -0.5
    * (1 - np.tanh((Rh2 - r0**2) / 0.2**2))
    * step_func(Z - d_forcing_b)
    * step_func(-(Z - (lz - d_forcing_b)))
)
variables["b"] = b

sim.state.init_statephys_from(**variables)

sim.state.statespect_from_statephys()
sim.state.statephys_from_statespect()

# monkey-patching for forcing
oper = sim.oper
X, Y, Z = oper.get_XYZ_loc()

alpha = step_func(d_forcing - Z) + step_func(Z - (lz - d_forcing))


# calculus of coef_sigma
"""
f(t) = f0 * exp(-sigma*t)

If we want f(t)/f0 = 10**(-gamma) after n_dt time steps, we have to have:

sigma = gamma / (n_dt * dt)
"""
gamma = 2
n_dt = 4
coef_sigma = gamma / n_dt


def compute_forcing_fft_each_time(self):
    """This function is called by the forcing_maker to compute the forcing"""

    def compute_forcing_1var(key):
        vi = self.sim.state.state_phys.get_var(key)
        sigma = coef_sigma / self.sim.time_stepping.deltat
        fi = -sigma * alpha * vi
        return oper.fft(fi)

    keys = ("vz",)
    result = {key + "_fft": compute_forcing_1var(key) for key in keys}
    return result


sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
    compute_forcing_fft_each_time
)

# finally we start the simulation
sim.time_stepping.start()

printby0(
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
