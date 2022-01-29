from fluidsim.solvers.ns3d.solver import Simul
import numpy as np

import fluiddyn as fld

params = Simul.create_default_params()

params.short_name_type_run = "test"

n = 64
L = 2 * np.pi
params.oper.nx = n
params.oper.ny = n
params.oper.nz = n
params.oper.Lx = L
params.oper.Ly = L
params.oper.Lz = L
params.oper.type_fft = "fluidfft.fft3d.mpi_with_fftwmpi3d"
# params.oper.type_fft = 'fluidfft.fft3d.with_cufft'

delta_x = params.oper.Lx / params.oper.nx
# params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
params.nu_8 = 2.0 * 10e-1 * delta_x**8

params.time_stepping.USE_T_END = True
params.time_stepping.t_end = 7.0
params.time_stepping.it_end = 2

# params.init_fields.type = 'dipole'

params.init_fields.type = "from_file"
params.init_fields.from_file.path = "/home/users/bonamy2c/Sim_data/ns3d_test_L=2pix2pix2pi_64x64x64_2016-07-27_20-02-21/state_phys_t=005.055.hd5"

params.forcing.enable = False
# params.forcing.type = 'random'
# 'Proportional'
# params.forcing.type_normalize

params.output.periods_print.print_stdout = 0.00000000001

params.output.periods_save.phys_fields = 1.0
# params.output.periods_save.spectra = 0.5
# params.output.periods_save.spatial_means = 0.05
# params.output.periods_save.spect_energy_budg = 0.5

# params.output.periods_plot.phys_fields = 0.0

params.output.ONLINE_PLOT_OK = True

# params.output.spectra.HAS_TO_PLOT_SAVED = True
# params.output.spatial_means.HAS_TO_PLOT_SAVED = True
# params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True

# params.output.phys_fields.field_to_plot = 'rot'

sim = Simul(params)

# sim.output.phys_fields.plot()
sim.time_stepping.start()
# sim.output.phys_fields.plot()

fld.show()
