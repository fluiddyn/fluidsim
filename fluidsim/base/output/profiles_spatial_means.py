import numpy as np
import h5py

from fluiddyn.util import mpi
from fluidsim.extend_simul import SimulExtender, extend_simul_class

from .base import SpecificOutput


__all__ = ["extend_simul_class", "ProfilesSpatialMeans"]


def _add_to_data_3d(arr3d, name: str, data: dict):
    mx = _add_to_data_x(arr3d, name, data)
    my = _add_to_data_y(arr3d, name, data)
    mz = _add_to_data_z(arr3d, name, data)
    return mx, my, mz


def _add_to_data_x(arr3d, name: str, data: dict):
    result = data[name + "_meanx"] = np.mean(arr3d, axis=(0, 1))
    return result


def _add_to_data_y(arr3d, name: str, data: dict):
    result = data[name + "_meany"] = np.mean(arr3d, axis=(0, 2))
    return result


def _add_to_data_z(arr3d, name: str, data: dict):
    result = data[name + "_meanz"] = np.mean(arr3d, axis=(1, 2))
    return result


class ProfilesSpatialMeans(SpecificOutput, SimulExtender):

    _tag = "profiles_spatial_means"
    _module_name = "fluidsim.base.output.profiles_spatial_means"
    _name_file = _tag + ".h5"

    @classmethod
    def get_modif_info_solver(cls):
        """Create a function to modify ``info_solver``.

        Note that this function is called when the object ``info_solver`` has
        not yet been created (and cannot yet be modified)! This is why one
        needs to create a function that will be called later to modify
        ``info_solver``.

        """

        def modif_info_solver(info_solver):

            info_solver.classes.Output.classes._set_child(
                cls._tag,
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def __init__(self, output):
        self.output = output
        params = output.sim.params

        dx = params.oper.Lx / params.oper.nx
        dy = params.oper.Ly / params.oper.ny
        dz = params.oper.Lz / params.oper.nz

        x = dx * np.arange(params.oper.nx)
        y = dy * np.arange(params.oper.ny)
        z = dz * np.arange(params.oper.nz)

        super().__init__(
            output,
            period_save=params.output.periods_save.profiles_spatial_means,
            arrays_1st_time={"x": x, "y": y, "z": z},
        )

    def compute(self):
        data = {}
        get_var = self.sim.state.state_phys.get_var

        vx = get_var("vx")
        vy = get_var("vy")
        vz = get_var("vz")

        vx_meanx, vx_meany, vx_meanz = _add_to_data_3d(vx, "vx", data)
        vy_meanx, vy_meany, vy_meanz = _add_to_data_3d(vy, "vy", data)
        vz_meanx, vz_meany, vz_meanz = _add_to_data_3d(vz, "vz", data)

        return data
