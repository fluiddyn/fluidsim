import numpy as np

from fluiddyn.util import mpi

from fluidsim.extend_simul import SimulExtender, extend_simul_class

from .base import SpecificOutput


__all__ = ["extend_simul_class", "ZProfilesSpatialMeans"]


class ZProfilesSpatialMeans(SpecificOutput, SimulExtender):

    _tag = "z_profiles_spatial_means"
    _module_name = "fluidsim.base.output.z_profiles_spatial_means"
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

    def _compute_z_profile(self, arr3d):
        if mpi.nb_proc == 1:
            return np.mean(arr3d, axis=(1, 2))

        raise NotImplementedError

    def _z_profile_to_3d_local(self, profile):
        if mpi.nb_proc == 1:
            return np.broadcast_to(
                profile[:, np.newaxis, np.newaxis], self.shapeX_loc
            )

        raise NotImplementedError

    def __init__(self, output):
        self.output = output
        params = output.sim.params

        self.shapeX_loc = output.sim.oper.shapeX_loc

        dz = params.oper.Lz / params.oper.nz
        z = dz * np.arange(params.oper.nz)

        super().__init__(
            output,
            period_save=params.output.periods_save.z_profiles_spatial_means,
            arrays_1st_time={"z": z},
        )

    def compute(self):
        data = {}
        get_var = self.sim.state.state_phys.get_var
        vx = get_var("vx")
        vy = get_var("vy")
        vz = get_var("vz")

        def _extend_data_with_z_profile(arr3d, name: str):
            result = data[name] = self._compute_z_profile(arr3d)
            return result

        vx_mean = _extend_data_with_z_profile(vx, "vx")
        vy_mean = _extend_data_with_z_profile(vy, "vy")
        vz_mean = _extend_data_with_z_profile(vz, "vz")

        vx_mean3d = self._z_profile_to_3d_local(vx_mean)
        vy_mean3d = self._z_profile_to_3d_local(vy_mean)
        vz_mean3d = self._z_profile_to_3d_local(vz_mean)

        vxp = vx - vx_mean3d
        vyp = vy - vy_mean3d
        vzp = vz - vz_mean3d

        _extend_data_with_z_profile(vxp**2, "vxp_vxp")
        _extend_data_with_z_profile(vyp**2, "vyp_vyp")
        _extend_data_with_z_profile(vzp**2, "vzp_vzp")

        _extend_data_with_z_profile(vyp * vxp, "vyp_vxp")
        _extend_data_with_z_profile(vzp * vxp, "vzp_vxp")
        _extend_data_with_z_profile(vzp * vyp, "vzp_vyp")

        return data
