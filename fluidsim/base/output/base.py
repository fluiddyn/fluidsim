"""Base module for the output (:mod:`fluidsim.base.output.base`)
================================================================

Provides:

.. autoclass:: OutputBase
   :members:
   :private-members:
   :noindex:

.. autoclass:: OutputBasePseudoSpectral
   :members:
   :private-members:

.. autoclass:: SpecificOutput
   :members:
   :private-members:

.. autoclass:: SimReprMaker
   :members:
   :private-members:
   :noindex:
"""

import datetime
import os
import shutil
import numbers
from copy import copy
from warnings import warn

import numpy as np
import h5py
import matplotlib.pyplot as plt

import fluiddyn
from fluiddyn.util import mpi
from fluiddyn.util import is_run_from_ipython, print_memory_usage
from fluiddyn.io import FLUIDSIM_PATH, Path
from fluidsim_core.output import OutputCore, SimReprMakerCore
from fluidsim_core.params import iter_complete_params

import fluidsim
from fluidsim.util.util import open_patient


class SimReprMaker(SimReprMakerCore):
    """Produce a string representing the simulation"""

    def get_time_as_str(self):
        params = self.sim.params
        if not params.NEW_DIR_RESULTS and (
            params.ONLY_COARSE_OPER or params.init_fields.type == "from_file"
        ):
            return self.time_from_path_run(params.path_run)
        else:
            return super().get_time_as_str()


class OutputBase(OutputCore):
    """Handle the output."""

    SimReprMaker = SimReprMaker

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        classes = info_solver.classes.Output._set_child("classes")

        classes._set_child(
            "PrintStdOut",
            attribs={
                "module_name": "fluidsim.base.output.print_stdout",
                "class_name": "PrintStdOutBase",
            },
        )

        classes._set_child(
            "PhysFields",
            attribs={
                "module_name": "fluidsim.base.output.phys_fields2d",
                "class_name": "PhysFieldsBase2D",
            },
        )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        attribs = {
            "ONLINE_PLOT_OK": True,
            "period_refresh_plots": 1,
            "HAS_TO_SAVE": True,
            "sub_directory": "",
        }
        p_output = params._set_child("output", attribs=attribs)

        p_output._set_doc(
            """

See :mod:`fluidsim.output.base`

ONLINE_PLOT_OK: bool (default: True)

    If True, the online plots are enabled.

period_refresh_plots: float (default: 1)

    Period of refreshment of the online plots.

HAS_TO_SAVE: bool (default: True)

    If False, nothing new is saved in the directory of the simulation.

sub_directory: str (default: "")

    A name of a subdirectory where the directory of the simulation is saved.
"""
        )

        p_output._set_child("periods_save")
        p_output.periods_save._set_doc(
            """
Periods (float, in equation time) to set when the specific outputs are saved.
"""
        )
        p_output._set_child("periods_print")
        p_output.periods_print._set_doc(
            """
Periods (float, in equation time) to set when the printing specific outputs are
called.
"""
        )
        p_output._set_child("periods_plot")
        p_output.periods_plot._set_doc(
            """
Periods (float, in equation time) to set when the plots of the specific outputs
are called.
"""
        )

        dict_classes = info_solver.classes.Output.import_classes()
        iter_complete_params(params, info_solver, dict_classes.values())

    def __init__(self, sim):
        super().__init__(sim)

        dict_classes = sim.info.solver.classes.Output.import_classes()

        PrintStdOut = dict_classes["PrintStdOut"]
        self.print_stdout = PrintStdOut(self)

        if not self.params.ONLINE_PLOT_OK:
            for k in self.params.periods_plot._get_key_attribs():
                self.params.periods_plot[k] = 0.0

        if not self._has_to_save:
            for k in self.params.periods_save._get_key_attribs():
                self.params.periods_save[k] = 0.0

    def _init_sim_repr_maker(self):
        sim_repr_maker = super()._init_sim_repr_maker()

        sim = self.sim
        # oper should already be initialized at this point
        if hasattr(self, "oper") and hasattr(self.oper, "_modify_sim_repr_maker"):
            self.oper._modify_sim_repr_maker(sim_repr_maker)

        if hasattr(sim, "_modify_sim_repr_maker"):
            sim._modify_sim_repr_maker(sim_repr_maker)

        # other classes are not initialized at this point
        dict_classes = sim.info_solver.import_classes()
        try:
            cls = dict_classes["Forcing"]
        except KeyError:
            pass
        else:
            if hasattr(cls, "_modify_sim_repr_maker"):
                cls._modify_sim_repr_maker(sim_repr_maker)

        return sim_repr_maker

    def init_with_oper_and_state(self):
        warn(
            "This method has been replaced with a more generic name `post_init`.",
            DeprecationWarning,
        )
        self.post_init()

    def post_init(self):
        sim = self.sim
        super().post_init()

        if mpi.rank == 0:

            objects_to_print = {
                "sim.oper": sim.oper,
                "sim.state": sim.state,
                "sim.time_stepping": sim.time_stepping,
                "sim.init_fields": sim.init_fields,
            }

            if hasattr(sim, "forcing"):
                objects_to_print["sim.forcing"] = sim.forcing

            if hasattr(sim, "preprocess"):
                objects_to_print["sim.preprocess"] = sim.preprocess

            for key, obj in objects_to_print.items():
                self.print_stdout(
                    "{:20s}".format(key + ": ") + str(obj.__class__)
                )

            # print info on the run
            if hasattr(sim.params.time_stepping, "type_time_scheme"):
                specifications = (
                    sim.params.time_stepping.type_time_scheme + " and "
                )
            else:
                specifications = ""
            if mpi.nb_proc == 1:
                specifications += "sequential,\n"
            else:
                specifications += f"parallel ({mpi.nb_proc} proc.)\n"
            self.print_stdout(
                "\nsolver "
                + self.name_solver
                + ", "
                + specifications
                + self.oper.produce_long_str_describing_oper()
                + "path_run =\n"
                + self.path_run
                + "\n"
                + "init_fields.type: "
                + sim.params.init_fields.type
                + "\n"
            )

            if hasattr(self.sim, "produce_str_describing_params"):
                self.print_stdout(
                    "Important parameters: \n"
                    + self.sim.produce_str_describing_params()
                )

        if mpi.rank == 0 and is_run_from_ipython():
            plt.ion()

        if sim.state.is_initialized:
            if hasattr(sim, "forcing") and not sim.forcing.is_initialized():
                return

            self.init_with_initialized_state()

    def _save_info_solver_params_xml(self, replace=False):
        """Save files with information on the solver and on the run."""
        if (
            mpi.rank == 0
            and self._has_to_save
            and self.sim.params.NEW_DIR_RESULTS
        ):
            super()._save_info_solver_params_xml(
                replace=replace,
                comment=f"FluidSim {fluidsim.get_local_version()}",
            )

    def init_with_initialized_state(self):

        if (
            hasattr(self, "_has_been_initialized_with_state")
            and self._has_been_initialized_with_state
        ):
            return

        else:
            self._has_been_initialized_with_state = True

        params = self.sim.params
        # just for the first output
        if (
            hasattr(params.time_stepping, "USE_CFL")
            and params.time_stepping.USE_CFL
        ):
            self.sim.time_stepping.compute_time_increment_CLF()

        if hasattr(self.sim, "forcing") and params.output.HAS_TO_SAVE:
            self.sim.forcing.compute()

        self.print_stdout("Initialization outputs:")

        self.print_stdout.complete_init_with_state()

        dict_classes = copy(self.sim.info.solver.classes.Output.import_classes())

        # The class PrintStdOut has already been instantiated.
        dict_classes.pop("PrintStdOut")

        # to get always the same order (important with mpi)
        keys = sorted(dict_classes.keys())
        classes = [dict_classes[key] for key in keys]

        for Class in classes:
            if mpi.rank == 0:
                self.print_stdout(
                    "{:30s}".format("sim.output." + Class._tag + ":") + str(Class)
                )
            self.__dict__[Class._tag] = Class(self)

        print_memory_usage("\nMemory usage at the end of init. (equiv. seq.)")

        try:
            self.print_size_in_Mo(self.sim.state.state_spect, "state_spect")
        except AttributeError:
            self.print_size_in_Mo(self.sim.state.state_phys, "state_phys")

    def one_time_step(self):

        for k in self.params.periods_print._get_key_attribs():
            period = self.params.periods_print.__dict__[k]
            if period != 0:
                self.__dict__[k]._online_print()

        if self.params.ONLINE_PLOT_OK:
            for k in self.params.periods_plot._get_key_attribs():
                period = self.params.periods_plot.__dict__[k]
                if period != 0:
                    self.__dict__[k]._online_plot()

        if self._has_to_save:
            for k in self.params.periods_save._get_key_attribs():
                period = self.params.periods_save.__dict__[k]
                if period != 0:
                    self.__dict__[k]._online_save()

    def figure_axe(self, numfig=None, size_axe=None):
        if mpi.rank == 0:
            if size_axe is None and numfig is None:
                return plt.subplots()

            if numfig is None:
                fig = plt.figure()
            else:
                fig = plt.figure(numfig)
                fig.clf()
            if size_axe is not None:
                ax = fig.add_axes(size_axe)
            else:
                ax = fig.subplots()
            return fig, ax

    def close_files(self):
        if mpi.rank == 0 and self._has_to_save:
            self.print_stdout.close()
            for k in self.params.periods_save._get_key_attribs():
                period = self.params.periods_save.__dict__[k]
                if period != 0:
                    if hasattr(self.__dict__[k], "_close_file"):
                        self.__dict__[k]._close_file()

    def end_of_simul(self, total_time):
        # self.path_run: str
        path_run = Path(self.path_run)
        self.print_stdout(
            f"Computation completed in {total_time:8.6g} s\n"
            f"path_run =\n{path_run}"
        )
        if self._has_to_save:
            if hasattr(self.sim, "forcing"):
                self.sim.forcing.compute()
            self.one_time_step()
            if self.sim.output.phys_fields.t_last_save < self.sim.time_stepping.t:
                self.phys_fields.save()

        self.close_files()

        if not self.path_run.startswith(FLUIDSIM_PATH):
            path_base = Path(FLUIDSIM_PATH)
            if len(self.params.sub_directory) > 0:
                path_base = path_base / self.params.sub_directory

            new_path_run = path_base / self.sim.name_run

            try:
                if new_path_run.parent.samefile(path_run.parent):
                    return
            except OSError:
                pass

            if mpi.rank == 0 and path_run.exists():
                if not path_base.exists():
                    os.makedirs(path_base)

                shutil.move(self.path_run, path_base)
                print(f"move result directory in directory:\n{new_path_run}")

            self.path_run = str(new_path_run)
            for spec_output in list(self.__dict__.values()):
                if isinstance(spec_output, SpecificOutput):
                    try:
                        spec_output._init_path_files()
                    except AttributeError:
                        pass

            if mpi.nb_proc > 1:
                mpi.comm.barrier()

    def compute_energy(self):
        return 0.0

    def print_size_in_Mo(self, arr, string=None):
        if string is None:
            string = "Size of ndarray (equiv. seq.)"
        else:
            string = "Size of " + string + " (equiv. seq.)"
        mem = arr.nbytes * 1.0e-6
        if mpi.nb_proc > 1:
            mem = mpi.comm.allreduce(mem, op=mpi.MPI.SUM)
        self.print_stdout(string.ljust(30) + f": {mem} Mo")


class OutputBasePseudoSpectral(OutputBase):
    def post_init(self):
        oper = self.oper
        self.sum_wavenumbers = oper.sum_wavenumbers
        super().post_init()

    def compute_energy_fft(self):
        """Compute energy(k)"""
        energy_fft = 0.0
        for k in self.sim.state.keys_state_spect:
            energy_fft += (
                np.abs(self.sim.state.state_spect.get_var(k)) ** 2
            ) / 2.0

        return energy_fft

    def compute_energy(self):
        """Compute the spatially averaged energy."""
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)


class SpecificOutput:
    """Small class for features useful for specific outputs"""

    def __init__(
        self,
        output,
        period_save=0,
        period_plot=0,
        has_to_plot_saved=False,
        arrays_1st_time=None,
    ):

        sim = output.sim
        params = sim.params

        self.output = output
        self.sim = sim
        self.oper = sim.oper
        self.params = params

        self.period_save = period_save
        self.period_plot = period_plot
        self.has_to_plot = has_to_plot_saved

        if not params.output.ONLINE_PLOT_OK:
            self.period_plot = 0
            self.has_to_plot = False

        if not has_to_plot_saved:
            if self.period_plot > 0:
                self.has_to_plot = True
            else:
                self.has_to_plot = False

        self.period_show = params.output.period_refresh_plots
        self.t_last_show = 0.0

        self._init_path_files()

        if self.has_to_plot:
            self._init_online_plot()

        if not output._has_to_save:
            self.period_save = 0.0

        if self.period_save != 0.0:
            self._init_files(arrays_1st_time)

    def _init_path_files(self):
        if hasattr(self, "_name_file"):
            self.path_file = os.path.join(self.output.path_run, self._name_file)

    def _init_files(self, arrays_1st_time=None):
        if arrays_1st_time is None:
            arrays_1st_time = {}
        dict_results = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                self._create_file_from_dict_arrays(
                    self.path_file, dict_results, arrays_1st_time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, "r") as file:
                    dset_times = file["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                self._add_dict_arrays_to_file(self.path_file, dict_results)
        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time."""
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_save >= self.period_save:
            self.t_last_save = tsim
            dict_results = self.compute()
            if mpi.rank == 0:
                self._add_dict_arrays_to_file(self.path_file, dict_results)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot_saving(dict_results)
                    if tsim - self.t_last_show >= self.period_show:
                        self.t_last_show = tsim
                        self.fig.canvas.draw()
                        # needed to really show the figures
                        plt.pause(1e-3)

    def _create_file_from_dict_arrays(
        self, path_file, dict_matrix, arrays_1st_time
    ):
        if os.path.exists(path_file):
            print("file NOT created since it already exists!")
        elif mpi.rank == 0:
            with h5py.File(path_file, "w") as file:
                file.attrs["date saving"] = str(datetime.datetime.now()).encode()
                file.attrs["name_solver"] = self.output.name_solver
                file.attrs["name_run"] = self.output.name_run

                self.sim.info._save_as_hdf5(hdf5_parent=file)

                times = np.array([self.sim.time_stepping.t], dtype=np.float64)
                file.create_dataset("times", data=times, maxshape=(None,))

                for k, v in list(arrays_1st_time.items()):
                    file.create_dataset(k, data=v)

                for k, v in list(dict_matrix.items()):
                    if isinstance(v, numbers.Number):
                        arr = np.array([v], dtype=v.__class__)
                        arr.resize((1,))
                        file.create_dataset(k, data=arr, maxshape=(None,))
                    else:
                        arr = np.array(v)
                        arr.resize((1,) + v.shape)
                        file.create_dataset(
                            k, data=arr, maxshape=((None,) + v.shape)
                        )

    def _add_dict_arrays_to_file(self, path_file, dict_matrix):
        if not os.path.exists(path_file):
            raise ValueError("can not add dict arrays in nonexisting file!")

        elif mpi.rank == 0:
            with open_patient(path_file, "r+") as file:
                dset_times = file["times"]
                nb_saved_times = dset_times.shape[0]
                dset_times.resize((nb_saved_times + 1,))
                dset_times[nb_saved_times] = self.sim.time_stepping.t
                for k, v in list(dict_matrix.items()):
                    if isinstance(v, numbers.Number):
                        dset_k = file[k]
                        dset_k.resize((nb_saved_times + 1,))
                        dset_k[nb_saved_times] = v
                    else:
                        dset_k = file[k]
                        dset_k.resize((nb_saved_times + 1,) + v.shape)
                        dset_k[nb_saved_times] = v

    def _add_dict_arrays_to_open_file(self, file, dict_arrays, nb_saved_times):
        if mpi.rank == 0:
            dset_times = file["times"]
            dset_times.resize((nb_saved_times + 1,))
            dset_times[nb_saved_times] = self.sim.time_stepping.t
            for k, v in list(dict_arrays.items()):
                dset_k = file[k]
                dset_k.resize((nb_saved_times + 1, v.size))
                dset_k[nb_saved_times] = v

    def _has_to_online_save(self):
        return (
            self.sim.time_stepping.t + 1e-15
        ) // self.period_save > self.t_last_save // self.period_save

    def _init_online_plot(self):
        pass

    def _online_plot_saving(self, dict_results):
        pass
