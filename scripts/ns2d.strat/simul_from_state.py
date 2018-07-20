"""
simul_from_state.py
===================
"""


import os
from glob import glob
from copy import deepcopy as _deepcopy
from fluidsim import load_sim_for_plot
from fluidsim.solvers.ns2d.strat.solver import Simul

# For gamma 0.2
path_root = "/fsnet/project/meige/2015/15DELDUCA/DataSim/Coef_Diss"
paths_gamma = glob(os.path.join(path_root, "Coef_Diss_gamma*"))

# gamma 0.2 - paths_gamma[0]
# gamma 0.5 - paths_gamma[1]
# gamma 1.0 - paths_gamma[2]
paths_sim = sorted(glob(paths_gamma[2] + "/NS2D.strat*"))

path_file = glob(paths_sim[-1] + "/state_phys*")[-1]


sim = load_sim_for_plot(paths_sim[-1])

params = _deepcopy(sim.params)

params.init_fields.type = "from_file"
params.init_fields.from_file.path = path_file

params.time_stepping.USE_CFL = False
params.time_stepping.t_end += 200
params.time_stepping.deltat0 = sim.time_stepping.deltat * 0.5

params.NEW_DIR_RESULTS = True

# params.oper.type_fft = "fft2d.mpi_with_fftw1d"

params.output.HAS_TO_SAVE = True
params.output.sub_directory = "sim480_stationarity"
params.output.periods_save.phys_fields = 2e-1
params.output.periods_save.spatial_means = 0.5
params.output.periods_save.spectra = 0.5
params.output.periods_save.spect_energy_budg = 0.5
params.output.periods_save.spatio_temporal_spectra = 1.

params.output.spatio_temporal_spectra.size_max_file = 100
params.output.spatio_temporal_spectra.spatial_decimate = 4


sim = Simul(params)
sim.time_stepping.start()
