import datetime

import numpy as np
import h5py
import h5netcdf

from fluiddyn.util import mpi

cfg_h5py = h5py.h5.get_config()

if cfg_h5py.mpi:
    ext = "h5"
    h5pack = h5py
else:
    ext = "nc"
    h5pack = h5netcdf


def _create_variable(group, key, field):
    if ext == "nc":
        if field.ndim == 0:
            dimensions = tuple()
        elif field.ndim == 1:
            dimensions = ("x",)
        elif field.ndim == 2:
            dimensions = ("y", "x")
        elif field.ndim == 3:
            dimensions = ("z", "y", "x")
        try:
            group.create_variable(key, data=field, dimensions=dimensions)
        except AttributeError:
            raise ValueError(
                "Error while creating a netCDF4 variable using group"
                f" of type {type(group)} for key {key}"
            )

    else:
        try:
            group.create_dataset(key, data=field)
        except AttributeError:
            raise ValueError(
                "Error while creating a HDF5 dataset using group"
                f" of type {type(group)} for key {key}"
            )


def save_file(
    path_file,
    state_phys,
    sim_info,
    output_name_run,
    oper,
    time,
    it,
    particular_attr=None,
):
    def create_group_with_attrs(h5file):
        group_state_phys = h5file.create_group("state_phys")
        group_state_phys.attrs["what"] = "obj state_phys for fluidsim"
        group_state_phys.attrs["name_type_variables"] = state_phys.info
        group_state_phys.attrs["time"] = time
        group_state_phys.attrs["it"] = it
        return group_state_phys

    if mpi.nb_proc == 1 or not cfg_h5py.mpi:
        if mpi.rank == 0:
            h5file = h5pack.File(str(path_file), "w")
            group_state_phys = create_group_with_attrs(h5file)
    else:
        h5file = h5pack.File(str(path_file), "w", driver="mpio", comm=mpi.comm)
        group_state_phys = create_group_with_attrs(h5file)

    if mpi.nb_proc == 1:
        for k in state_phys.keys:
            field_seq = state_phys.get_var(k)
            _create_variable(group_state_phys, k, field_seq)
    elif not cfg_h5py.mpi:
        for k in state_phys.keys:
            field_loc = state_phys.get_var(k)
            field_seq = oper.gather_Xspace(field_loc)
            if mpi.rank == 0:
                _create_variable(group_state_phys, k, field_seq)
    else:
        h5file.atomic = False
        ndim = len(oper.shapeX_loc)
        if ndim == 2:
            xstart, ystart = oper.seq_indices_first_X
        elif ndim == 3:
            xstart, ystart, zstart = oper.seq_indices_first_X
        else:
            raise NotImplementedError
        xend = xstart + oper.shapeX_loc[0]
        yend = ystart + oper.shapeX_loc[1]
        for k in state_phys.keys:
            field_loc = state_phys.get_var(k)
            dset = group_state_phys.create_dataset(
                k, oper.shapeX_seq, dtype=field_loc.dtype
            )
            with dset.collective:
                if field_loc.ndim == 2:
                    dset[xstart:xend, ystart:yend] = field_loc
                elif field_loc.ndim == 3:
                    dset[xstart:xend, ystart:yend, :] = field_loc
                else:
                    raise NotImplementedError("Unsupported number of dimensions")
        h5file.close()
        if mpi.rank == 0:
            h5file = h5pack.File(str(path_file), "r+")

    if mpi.rank == 0:
        h5file.attrs["date saving"] = str(datetime.datetime.now()).encode()
        h5file.attrs["name_solver"] = sim_info.solver.short_name
        h5file.attrs["name_run"] = output_name_run
        h5file.attrs["axes"] = np.array(oper.axes, dtype="|S9")
        if particular_attr is not None:
            h5file.attrs["particular_attr"] = particular_attr

        sim_info._save_as_hdf5(hdf5_parent=h5file)
        gp_info = h5file["info_simul"]
        gf_params = gp_info["params"]
        gf_params.attrs["SAVE"] = 1
        gf_params.attrs["NEW_DIR_RESULTS"] = 1
        h5file.close()
