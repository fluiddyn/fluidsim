"""Initialisation of the fields (:mod:`fluidsim.base.init_fields`)
========================================================================

Provides:

.. autoclass:: InitFieldsBase
   :members:
   :private-members:

"""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import h5py
import h5netcdf

from fluiddyn.util import mpi
from fluidsim_core.params import iter_complete_params

from fluidsim.base.setofvariables import SetOfVariables


class InitFieldsBase:
    """Initialization of the fields (base class)."""

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        """Static method to complete the ParamContainer info_solver.

        Parameters
        ----------

        info_solver : fluiddyn.util.paramcontainer.ParamContainer

        classes : iterable of classes (SpecificInitFields)

          If a class has the same tag of a default class, it replaces the
          default one (for example with the tag 'noise').

        """
        classesXML = info_solver.classes.InitFields._set_child("classes")

        classes_used = [
            InitFieldsFromFile,
            InitFieldsFromSimul,
            InitFieldsInScript,
            InitFieldsConstant,
            InitFieldsNoise,
        ]

        classes_used = {cls.tag: cls for cls in classes_used}

        if classes is not None:
            for cls in classes:
                classes_used[cls.tag] = cls

        for tag, cls in classes_used.items():
            classesXML._set_child(
                tag,
                attribs={
                    "module_name": cls.__module__,
                    "class_name": cls.__name__,
                },
            )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        params._set_child(
            "init_fields",
            attribs={
                "type": "constant",
                "available_types": [],
                "modif_after_init": False,
            },
        )

        params.init_fields._set_doc(
            """

See :mod:`fluidsim.base.init_fields`.

type: str (default constant)

    Name of the initialization method.

available_types: list

    Actually not a parameter; just a hint to set `type`.

modif_after_init: bool (default False)

    Used internally when reloading some simulations.

"""
        )

        dict_classes = info_solver.classes.InitFields.import_classes()
        iter_complete_params(params, info_solver, dict_classes.values())

    def __init__(self, sim):

        self.sim = sim
        params = sim.params
        oper = sim.oper

        self.params = params
        self.oper = oper

        type_init = params.init_fields.type
        if type_init not in params.init_fields.available_types:
            raise ValueError(
                type_init + " is not an available flow initialization."
            )

        dict_classes = sim.info.solver.classes.InitFields.import_classes()
        Class = dict_classes[type_init]
        self._specific_init_fields = Class(sim)

    def __call__(self):
        self.sim.state.is_initialized = not bool(
            self.params.init_fields.modif_after_init
        )
        self._specific_init_fields()


class SpecificInitFields:
    tag = "specific"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.init_fields.available_types.append(cls.tag)

    def __init__(self, sim):
        self.sim = sim


class InitFieldsFromFile(SpecificInitFields):

    tag = "from_file"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={"path": ""})
        params.init_fields.from_file._set_doc(
            """
path: str
"""
        )

    def __call__(self):

        # Warning: this function is for 2d pseudo-spectral solver!
        # We have to write something more general.

        params = self.sim.params

        path_file = params.init_fields.from_file.path
        if isinstance(path_file, Path):
            path_file = str(path_file)

        if mpi.rank == 0:
            try:
                if os.path.splitext(path_file)[1] == ".nc":
                    h5file = h5netcdf.File(path_file, "r")
                else:
                    h5file = h5py.File(path_file, "r")
            except Exception:
                raise ValueError(
                    "Is file " + path_file + " really a netCDF4/HDF5 file?"
                )

            print("Load state from file:\n[...]" + path_file[-75:])

            try:
                group_oper = h5file["/info_simul/params/oper"]
            except Exception:
                raise ValueError(
                    "The file " + path_file + " does not contain a params object"
                )

            try:
                group_state_phys = h5file["/state_phys"]
            except Exception:
                raise ValueError(
                    "The file "
                    + path_file
                    + " does not contain a state_phys object"
                )

            try:
                axes = h5file.attrs["axes"]
                for r in axes:
                    # for example r can be: 'z', 'y', 'x'
                    r = r.decode("utf-8")
                    nr = f"n{r}"
                    nr_file = group_oper.attrs[nr]
                    if params.oper[nr] != nr_file:
                        raise ValueError(
                            "this is not a correct state for this simulation\n"
                            "self.{0} != params_file.{0}".format(nr)
                        )
                    Lr = f"L{r}"
                    try:
                        Lr_file = group_oper.attrs[Lr]
                    except KeyError:
                        # Length may not be a parameter for eg: sphericalharmo
                        continue
                    else:
                        if params.oper[Lr] != Lr_file:
                            raise ValueError(
                                "this is not a correct state for this simulation\n"
                                "self.params.oper.{0} != params_file.{0}".format(
                                    Lr
                                )
                            )
            except KeyError:
                # Legacy purposes: 2D specific
                nx_file = group_oper.attrs["nx"]
                ny_file = group_oper.attrs["ny"]
                Lx_file = group_oper.attrs["Lx"]
                Ly_file = group_oper.attrs["Ly"]

                if isinstance(nx_file, list):
                    nx_file = nx_file.item()
                    ny_file = ny_file.item()
                    Lx_file = Lx_file.item()
                    Ly_file = Ly_file.item()

                if params.oper.nx != nx_file:
                    raise ValueError(
                        "this is not a correct state for this simulation\n"
                        "self.nx != params_file.nx"
                    )

                if params.oper.ny != ny_file:
                    raise ValueError(
                        "this is not a correct state for this simulation\n"
                        "self.ny != params_file.ny"
                    )

                if params.oper.Lx != Lx_file:
                    raise ValueError(
                        "this is not a correct state for this simulation\n"
                        "self.params.oper.Lx != params_file.Lx"
                    )

                if params.oper.Ly != Ly_file:
                    raise ValueError(
                        "this is not a correct state for this simulation\n"
                        "self.params.oper.Ly != params_file.Ly"
                    )

            keys_state_phys_file = list(group_state_phys.keys())
        else:
            keys_state_phys_file = {}
        if mpi.nb_proc > 1:
            keys_state_phys_file = mpi.comm.bcast(keys_state_phys_file)
        state_phys = self.sim.state.state_phys
        keys_phys_needed = self.sim.info.solver.classes.State.keys_phys_needed
        for k in keys_phys_needed:
            if k in keys_state_phys_file:
                if mpi.rank == 0:
                    field_seq = group_state_phys[k][...]
                else:
                    field_seq = None

                if mpi.nb_proc > 1:
                    field_loc = self.sim.oper.scatter_Xspace(field_seq)
                else:
                    field_loc = field_seq
                state_phys.set_var(k, field_loc)
            else:
                state_phys.set_var(k, self.sim.oper.create_arrayX(value=0.0))
        if mpi.rank == 0:
            time = group_state_phys.attrs["time"]
            try:
                it = group_state_phys.attrs["it"]
            except KeyError:
                # compatibility with older versions
                it = 0
            h5file.close()
        else:
            time = 0.0
            it = 0

        if mpi.nb_proc > 1:
            time = mpi.comm.bcast(time)
            it = mpi.comm.bcast(it)

        if hasattr(self.sim.state, "statespect_from_statephys"):
            self.sim.state.statespect_from_statephys()
            self.sim.state.statephys_from_statespect()
        self.sim.time_stepping.t = time
        self.sim.time_stepping.it = it


def fill_field_fft_2d(field_fft_in, field_fft_out):

    [nk0_seq, nk1_seq] = field_fft_out.shape
    [nk0_seq_in, nk1_seq_in] = field_fft_in.shape

    nk0_min = min(nk0_seq, nk0_seq_in)
    nk1_min = min(nk1_seq, nk1_seq_in)

    # it is a little bit complicate to take into account ky
    for ik1 in range(nk1_min):
        field_fft_out[0, ik1] = field_fft_in[0, ik1]
        field_fft_out[nk0_min // 2, ik1] = field_fft_in[nk0_min // 2, ik1]
    for ik0 in range(1, nk0_min // 2):
        for ik1 in range(nk1_min):
            field_fft_out[ik0, ik1] = field_fft_in[ik0, ik1]
            field_fft_out[-ik0, ik1] = field_fft_in[-ik0, ik1]


def fill_field_fft_3d(field_fft_in, field_fft_out, oper_in, oper_out):

    [nk0, nk1, nk2] = field_fft_out.shape
    [nk0_in, nk1_in, nk2_in] = field_fft_in.shape

    nk0_min = min(nk0, nk0_in)
    nk1_min = min(nk1, nk1_in)
    nk2_min = min(nk2, nk2_in)

    for ik0 in range(nk0_min):
        for ik1 in range(nk1_min):
            for ik2 in range(nk2_min):
                kx_adim, ky_adim, kz_adim = oper_in.kadim_from_ik012rank(
                    ik0, ik1, ik2
                )
                oper_out.set_value_spect(
                    field_fft_out,
                    field_fft_in[ik0, ik1, ik2],
                    kx_adim,
                    ky_adim,
                    kz_adim,
                )


class InitFieldsFromSimul(SpecificInitFields):

    tag = "from_simul"

    def __call__(self):
        self.sim.init_fields.get_state_from_simul = self._get_state_from_simul

    def _make_state_spect_2d(self, sim_in):
        sim = self.sim
        if (
            sim.params.oper.nx == sim_in.params.oper.nx
            and sim.params.oper.ny == sim_in.params.oper.ny
        ):
            return deepcopy(sim_in.state.state_spect)

        # modify resolution
        state_spect = SetOfVariables(like=sim.state.state_spect, value=0.0)
        keys_state_spect = sim_in.info.solver.classes.State["keys_state_spect"]
        for index_key in range(len(keys_state_spect)):
            field_fft_in = sim_in.state.state_spect[index_key]
            field_fft_new_res = state_spect[index_key]
            fill_field_fft_2d(field_fft_in, field_fft_new_res)

        return state_spect

    def _make_state_spect_3d(self, sim_in):
        # sim = self.sim
        sim = self.sim
        oper_in = sim_in.oper

        if (
            sim.params.oper.nx == sim_in.params.oper.nx
            and sim.params.oper.ny == sim_in.params.oper.ny
            and sim.params.oper.nz == sim_in.params.oper.nz
        ):
            return deepcopy(sim_in.state.state_spect)

        # modify resolution
        state_spect = SetOfVariables(like=sim.state.state_spect, value=0.0)
        keys_state_spect = sim_in.info.solver.classes.State["keys_state_spect"]
        for index_key in range(len(keys_state_spect)):
            field_fft_in = sim_in.state.state_spect[index_key]
            field_fft_new_res = state_spect[index_key]
            fill_field_fft_3d(field_fft_in, field_fft_new_res, oper_in, sim.oper)

        return state_spect

    def _get_state_from_simul(self, sim_in):

        # Warning: this function is for 2d pseudo-spectral solver!
        # We have to write something more general.
        # It should be done directly in the operators.

        if mpi.nb_proc > 1:
            raise NotImplementedError(
                "THIS METHOD WON'T BE IMPLEMENTED IN MPI. "
                "The resolution has to be modified in sequential."
            )

        sim = self.sim
        sim.time_stepping.t = sim_in.time_stepping.t

        try:
            sim.params.oper.ny
            try:
                sim.params.oper.nz
            except AttributeError:
                nb_dim = 2
            else:
                sim_in.params.oper.nz
                nb_dim = 3
        except AttributeError:
            nb_dim = 1
        else:
            sim_in.params.oper.ny

        if nb_dim == 1:
            raise NotImplementedError()
        elif nb_dim == 2:
            state_spect = self._make_state_spect_2d(sim_in)
        elif nb_dim == 3:
            state_spect = self._make_state_spect_3d(sim_in)

        if sim.output.name_solver == sim_in.output.name_solver:
            sim.state.state_spect = state_spect
        else:  # complicated case... untested solution !
            raise NotImplementedError
            # state_spect = SetOfVariables('state_spect')
            # for k in sim.info.solver.classes.State["keys_state_spect"]:
            #     if k in sim_in.info.solver.classes.State["keys_state_spect"]:
            #         sim.state.state_spect[k] = state_spect[k]
            #     else:
            #         sim.state.state_spect[k] = self.oper.create_arrayK(value=0.0)

        sim.state.statephys_from_statespect()


class InitFieldsInScript(SpecificInitFields):

    tag = "in_script"

    def __call__(self):
        self.sim.state.is_initialized = False
        self.sim.output.print_stdout(
            "Manual initialization of the fields is selected. "
            "Do not forget to initialize them."
        )


class InitFieldsConstant(SpecificInitFields):

    tag = "constant"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={"value": 1.0})
        params.init_fields.constant._set_doc(
            """
value: float (default 1.)
"""
        )

    def __call__(self):
        value = self.sim.params.init_fields.constant.value
        self.sim.state.state_phys.initialize(value)

        if hasattr(self.sim.state, "statespect_from_statephys"):
            self.sim.state.statespect_from_statephys()


class InitFieldsNoise(SpecificInitFields):
    """Initialize the state with noise."""

    tag = "noise"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={"max": 1.0})
        params.init_fields.noise._set_doc(
            """
max: float (default 1.)
"""
        )

    def __call__(self):
        state_phys = self.sim.state.state_phys
        state_phys[...] = (
            self.sim.params.init_fields.noise.max
            / 0.5
            * (np.random.rand(*state_phys.shape) - 0.5)
        )

        if hasattr(self.sim.state, "statespect_from_statephys"):
            self.sim.state.statespect_from_statephys()
            self.sim.state.statephys_from_statespect()
