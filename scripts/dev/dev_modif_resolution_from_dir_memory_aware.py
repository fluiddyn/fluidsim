from copy import deepcopy as _deepcopy
from pathlib import Path

import h5py
import h5netcdf

from fluidsim.solvers.ns2d.solver import Simul as _Simul
from fluidsim.util.util import modif_resolution_from_dir
from fluidsim.util.util import (
    pathdir_from_namedir,
    _import_solver_from_path,
    load_params_simul,
    name_file_from_time_approx,
)

from fluidsim.base.solvers.info_base import create_info_simul
from fluidsim.base.init_fields import fill_field_fft_2d, fill_field_fft_3d
from fluidsim.base.output.phys_fields import save_file

params = _Simul.create_default_params()

params.oper.nx = 64
params.oper.ny = 32
params.init_fields.type = "noise"

sim = _Simul(params)
sim.output.phys_fields.save()
sim.output.close_files()

path_run = name_dir = sim.output.path_run
# now we don't need sim and params
# even we have to get them from the directory
del sim, params

# input parameters of the modif_resol functions
coef_modif_resol = 3 / 2
t_approx = None

# first we use the old function
modif_resolution_from_dir(
    name_dir, t_approx=t_approx, coef_modif_resol=coef_modif_resol, PLOT=0
)
path_new_file = next(Path(path_run).glob("State_phys_*/state_phys*"))
path_new_file = path_new_file.rename(path_new_file.with_name("old_" + path_new_file.name))

# Then, the alternative implementation
path_dir = pathdir_from_namedir(name_dir)

solver = _import_solver_from_path(path_dir)

Simul = solver.Simul

classes = Simul.info_solver.import_classes()
Operators = classes["Operators"]

params = load_params_simul(path_dir)


oper = Operators(params=params)

params2 = _deepcopy(params)
params2.output.HAS_TO_SAVE = True
params2.oper.nx = int(params.oper.nx * coef_modif_resol)
params2.oper.ny = int(params.oper.ny * coef_modif_resol)

try:
    params2.oper.nz = int(params.oper.nz * coef_modif_resol)
except AttributeError:
    dimension = 2
else:
    dimension = 3

oper2 = Operators(params=params2)
info2 = create_info_simul(Simul.info_solver, params2)


class StatePhysLike:
    def __init__(self, path_file, oper, oper2):
        self.path_file = path_file
        self.oper = oper
        self.oper2 = oper2
        self.info = "state_phys"

        if path_file.suffix == ".nc":
            self.h5pack = h5netcdf
        else:
            self.h5pack = h5py

        with self.h5pack.File(self.path_file, "r") as h5file:
            group_state_phys = h5file["/state_phys"]
            self.keys = list(group_state_phys.keys())
            self.time = float(group_state_phys.attrs["time"])
            self.it = int(group_state_phys.attrs["it"])
            self.name_run = h5file.attrs["name_run"]

    def get_var(self, key):

        with self.h5pack.File(self.path_file, "r") as h5file:
            group_state_phys = h5file["/state_phys"]
            field = group_state_phys[key][...]

        field_spect = self.oper.fft(field)

        dimension = len(field_spect.shape)
        if dimension not in [2, 3]:
            raise NotImplementedError

        field_spect_new = self.oper2.create_arrayK(0)

        if dimension == 2:
            fill_field_fft_2d(field_spect, field_spect_new)
        else:
            fill_field_fft_3d(field_spect, field_spect_new, self.oper, self.oper2)

        return self.oper2.ifft(field_spect_new)


name_file = name_file_from_time_approx(path_dir, t_approx)
path_file = path_dir / name_file

state_phys = StatePhysLike(path_file, oper, oper2)


if dimension == 3:
    dir_new_new = f"State_phys_{oper2.nx}x{oper2.ny}x{oper2.nz}"
else:
    dir_new_new = f"State_phys_{oper2.nx}x{oper2.ny}"

path_file_out = path_file.parent / dir_new_new / path_file.name

save_file(
    path_file_out,
    state_phys,
    info2,
    state_phys.name_run,
    oper2,
    state_phys.time,
    state_phys.it,
    particular_attr="modif_resolution",
)
