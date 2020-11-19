from pathlib import Path

import numpy as np
import h5py

from fluidsim.solvers.ns3d.solver import Simul
from fluidsim.util.util import (
    modif_resolution_from_dir,
    modif_resolution_from_dir_memory_efficient,
)

params = Simul.create_default_params()

params.output.sub_directory = "dev"

params.oper.nx = 24
params.oper.ny = 12
params.oper.nz = 12
params.init_fields.type = "noise"

sim = Simul(params)
sim.output.phys_fields.save()
sim.output.close_files()

path_run = sim.output.path_run
del sim, params

# input parameters of the modif_resol functions
t_approx = None
coef_modif_resol = 3 / 2

# first, the standard function
modif_resolution_from_dir(path_run, t_approx, coef_modif_resol, PLOT=False)
path_big = next(Path(path_run).glob("State_phys_*/state_phys*"))
path_big_old = path_big.rename(path_big.with_name("old_" + path_big.name))

# Then, the alternative implementation
modif_resolution_from_dir_memory_efficient(path_run, t_approx, coef_modif_resol)

with h5py.File(path_big_old, "r") as file:
    group_state_phys = file["/state_phys"]
    key = list(group_state_phys.keys())[0]
    field_old = group_state_phys[key][...]

with h5py.File(path_big, "r") as file:
    group_state_phys = file["/state_phys"]
    field = group_state_phys[key][...]

assert np.allclose(field_old, field)
