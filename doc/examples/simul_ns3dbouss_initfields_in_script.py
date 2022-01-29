"""Script for a short simulation with the solver ns3d.bouss

The field initialization is done in the script.

Launch with::

  mpirun -np 4 python simul_ns3dbouss_initfields_in_script.py

"""

import numpy as np

from fluiddyn.util.mpi import printby0

from fluidsim.solvers.ns3d.bouss.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = "examples"

nx = 48 // 2
ny = 64 // 2
nz = 96 // 2
Lx = 3
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = Lx
params.oper.Ly = Ly = Lx / nx * ny
params.oper.Lz = Lz = Lx / nx * nz
fft = "fftwmpi3d"
# fft = 'fftw1d'
# fft = 'pfft'
# fft = 'p3dfft'

# for sequential runs, just comment these 2 lines
params.oper.type_fft = "fluidfft.fft3d.mpi_with_" + fft
params.short_name_type_run = fft


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
dx = Lx / nx
B = 1
D = 1
eps = 1e-2 * B ** (3 / 2) * D ** (1 / 2)
params.nu_8 = (dx / C) ** ((3 * n - 2) / 3) * eps ** (1 / 3)

printby0(f"nu_8 = {params.nu_8:.3e}")

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 10.0

params.init_fields.type = "in_script"

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5
params.output.periods_save.spatial_means = 0.1


sim = Simul(params)

# here we have to initialize the flow fields

variables = {
    k: 1e-2 * sim.oper.create_arrayX_random() for k in ("vx", "vy", "vz")
}

X, Y, Z = sim.oper.get_XYZ_loc()

x0 = Lx / 2.0
y0 = Ly / 2.0
z0 = Lz / 2.0
R2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
r0 = 0.5
b = -0.5 * (1 - np.tanh((R2 - r0**2) / 0.2**2))
variables["b"] = b

sim.state.init_statephys_from(**variables)

sim.state.statespect_from_statephys()
sim.state.statephys_from_statespect()

sim.time_stepping.start()

printby0(
    f"""
To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

"""
)
