"""Script for a short simulation with the solver ns3d.bouss

The field initialization is done in the script.

Launch with::

  mpirun -np 4 python simul_ns3dbouss_staticlayer.py

"""

import numpy as np

from fluiddyn.util.mpi import printby0, rank

from fluidsim.solvers.ns3d.bouss.solver import Simul

params = Simul.create_default_params()

params.output.sub_directory = 'examples'

nx = 96
ny = 64
nz = 48
lz = 2
params.oper.nx = nx
params.oper.ny = ny
params.oper.nz = nz
params.oper.Lx = lx = lz/nz*nx
params.oper.Ly = ly = lz/nz*ny
params.oper.Lz = lz
fft = 'fftwmpi3d'  # works fine
fft = 'fftw1d'  # works fine
# fft = 'pfft'  # gather buggy
fft = 'p3dfft'  # gather buggy

# for sequential runs, just comment this line
# params.oper.type_fft = 'fluidfft.fft3d.mpi_with_' + fft

# params.short_name_type_run = fft

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
C = 2.
dx = lx/nx
B = 1
D = 1
eps = 1e-2*B**(3/2)*D**(1/2)
params.nu_8 = (dx/C)**((3*n-2)/3) * eps**(1/3)

# printby0(f'nu_8 = {params.nu_8:.3e}')

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 8.
# we need small time step for a strong forcing
params.time_stepping.USE_CFL = False
params.time_stepping.deltat0 = 0.015

params.init_fields.type = 'in_script'

params.forcing.enable = True
params.forcing.type = 'in_script'
params.forcing.nkmax_forcing = 20

params.output.periods_print.print_stdout = 1e-1

params.output.periods_save.phys_fields = 0.5

sim = Simul(params)

# here we have to initialize the flow fields

variables = {k: 1e-6 * sim.oper.create_arrayX_random()
             for k in ('vx', 'vy', 'vz')}

X, Y, Z = sim.oper.get_XYZ_loc()

x0 = lx/2.
y0 = ly/2.
z0 = lz/2.
R2 = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
r0 = 0.5
b = -0.5*(1-np.tanh((R2 - r0**2)/0.2**2))
variables['b'] = b

sim.state.init_statephys_from(**variables)

sim.state.statespect_from_statephys()
sim.state.statephys_from_statespect()


# monkey-patching for forcing
if rank == 0:

    forcing_maker = sim.forcing.forcing_maker
    oper = forcing_maker.oper_coarse
    X, Y, Z = oper.get_XYZ_loc()

    d = lz/6
    alpha = - 80 * (np.exp(-Z**2/d**2) + np.exp(-(Z - lz)**2/d**2))
    
    def compute_forcingc_fft_each_time(self):
        """This function is called by the forcing_maker to compute the forcing

        """
        def compute_forcing_1var(key):
            vi_fft = self.sim.state.state_spect.get_var(key + '_fft')
            vi_fft = self.oper.coarse_seq_from_fft_loc(
                vi_fft, self.shapeK_loc_coarse)
            vi = oper.ifft(vi_fft)
            fi = alpha * vi
            return oper.fft(fi)

        # keys = ('vx', 'vy', 'vz')
        keys = ('vz',)
        result = {key + '_fft': compute_forcing_1var(key)
                  for key in keys}
        return result

    forcing_maker.monkeypatch_compute_forcingc_fft_each_time(
        compute_forcingc_fft_each_time)


sim.time_stepping.start()

printby0(f"""
To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

""")
