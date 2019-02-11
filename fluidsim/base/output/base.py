"""Base module for the output (:mod:`fluidsim.base.output.base`)
================================================================

Provides:

.. autoclass:: OutputBase
   :members:
   :private-members:

.. autoclass:: OutputBasePseudoSpectral
   :members:
   :private-members:

.. autoclass:: SpecificOutput
   :members:
   :private-members:

"""

import datetime
import os
import shutil
import numbers
from time import sleep

import numpy as np
import h5py
import matplotlib.pyplot as plt

import fluiddyn
from fluiddyn.util import mpi
from fluiddyn.util import is_run_from_ipython, time_as_str, print_memory_usage
from fluiddyn.io import FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH, Path

import fluidsim
from fluidsim.util.util import load_params_simul


class OutputBase:
    """Handle the output."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        info_solver.classes.Output._set_child("classes")
        classes = info_solver.classes.Output.classes

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
        """This static method is used to complete the *params* container.
        """
        attribs = {
            "ONLINE_PLOT_OK": True,
            "period_refresh_plots": 1,
            "HAS_TO_SAVE": True,
            "sub_directory": "",
        }
        params._set_child("output", attribs=attribs)

        params.output._set_doc(
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

        params.output._set_child("periods_save")
        params.output.periods_save._set_doc(
            """
Periods (float, in equation time) to set when the specific outputs are saved.
"""
        )
        params.output._set_child("periods_print")
        params.output.periods_print._set_doc(
            """
Periods (float, in equation time) to set when the printing specific outputs are
called.
"""
        )
        params.output._set_child("periods_plot")
        params.output.periods_plot._set_doc(
            """
Periods (float, in equation time) to set when the plots of the specific outputs
are called.
"""
        )

        dict_classes = info_solver.classes.Output.import_classes()
        for Class in list(dict_classes.values()):
            if hasattr(Class, "_complete_params_with_default"):
                try:
                    Class._complete_params_with_default(params)
                except TypeError:
                    try:
                        Class._complete_params_with_default(params, info_solver)
                    except TypeError as e:
                        e.args += ("for class: " + repr(Class),)
                        raise

    def __init__(self, sim):
        params = sim.params
        self.sim = sim
        self.params = params.output
        self.oper = sim.oper

        self._has_to_save = self.params.HAS_TO_SAVE
        self.name_solver = sim.info.solver.short_name

        # initialisation name_run and path_run
        self._init_name_run()
        if mpi.nb_proc > 1:
            # ensure same name_run across all processes
            self.name_run = mpi.comm.bcast(self.name_run, root=0)

        self.sim.name_run = self.name_run

        if not params.NEW_DIR_RESULTS:
            try:
                self.path_run = str(params.path_run)
            except AttributeError:
                params.NEW_DIR_RESULTS = True
                print(
                    "Strange: params.NEW_DIR_RESULTS == False "
                    "but no params.path_run"
                )

            # if _has_to_save, we verify the correspondence between the
            # resolution of the simulation and the resolution of the
            # previous simulation saved in this directory
            if self._has_to_save:
                if mpi.rank == 0:
                    try:
                        params_dir = load_params_simul(self.path_run)
                    except:
                        raise ValueError(
                            "Strange, no info_simul.h5 in self.path_run"
                        )

                    cond = False
                    try:
                        if params.oper.nx != params_dir.oper.nx:
                            cond = True
                    except AttributeError:
                        pass
                    try:
                        if params.oper.ny != params_dir.oper.ny:
                            cond = True
                    except AttributeError:
                        pass
                    try:
                        if params.oper.nz != params_dir.oper.nz:
                            cond = True
                    except AttributeError:
                        pass

                    if cond:
                        params.NEW_DIR_RESULTS = True
                        print(
                            """
Warning: params.NEW_DIR_RESULTS is False but the resolutions of the simulation
         and of the simulation in the directory self.path_run are different
         we put params.NEW_DIR_RESULTS = True"""
                        )
                if mpi.nb_proc > 1:
                    params.NEW_DIR_RESULTS = mpi.comm.bcast(
                        params.NEW_DIR_RESULTS
                    )

        if params.NEW_DIR_RESULTS:

            if FLUIDDYN_PATH_SCRATCH is not None:
                path_base = FLUIDDYN_PATH_SCRATCH
            else:
                path_base = FLUIDSIM_PATH

            if len(params.output.sub_directory) > 0:
                path_base = os.path.join(path_base, params.output.sub_directory)

            if mpi.rank == 0:
                while True:
                    path_run = os.path.join(path_base, self.name_run)
                    if not os.path.exists(path_run):
                        try:
                            os.makedirs(path_run)
                        except OSError:
                            # in case of simultaneously launched simulations
                            print(
                                'Warning: NEW_DIR_RESULTS=True, but path"',
                                path_run,
                                "already exists. Trying a new path...",
                            )
                            sleep(1)
                            self._init_name_run()
                        else:
                            break

            else:
                path_run = ""

            if mpi.nb_proc > 1:
                self.path_run = mpi.comm.bcast(path_run, root=0)
                self.name_run = mpi.comm.bcast(self.name_run, root=0)
            else:
                self.path_run = path_run

            self.sim.name_run = self.name_run

            if mpi.rank == 0:
                params._set_attrib("path_run", self.path_run)

        dict_classes = sim.info.solver.classes.Output.import_classes()

        PrintStdOut = dict_classes["PrintStdOut"]
        self.print_stdout = PrintStdOut(self)

        if not self.params.ONLINE_PLOT_OK:
            for k in self.params.periods_plot._get_key_attribs():
                self.params.periods_plot[k] = 0.0

        if not self._has_to_save:
            for k in self.params.periods_save._get_key_attribs():
                self.params.periods_save[k] = 0.0

    def _init_name_run(self):
        """Initialize the run name"""
        list_for_name_run = self._create_list_for_name_run()
        list_for_name_run.append(time_as_str())
        self.name_run = "_".join(list_for_name_run)

    def _create_list_for_name_run(self):
        """Create a list of strings to make the run name."""
        list_for_name_run = [self.name_solver]
        if len(self.sim.params.short_name_type_run) > 0:
            list_for_name_run.append(self.sim.params.short_name_type_run)
        if hasattr(self, "oper"):
            str_describing_oper = self.oper.produce_str_describing_oper()
            if len(str_describing_oper) > 0:
                list_for_name_run.append(str_describing_oper)

        return list_for_name_run

    def init_with_oper_and_state(self):
        sim = self.sim

        if mpi.rank == 0:

            objects_to_print = {
                "sim": sim,
                "sim.oper": sim.oper,
                "sim.output": sim.output,
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

        self._save_info_solver_params_xml()

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
            comment = (
                "This file has been created by"
                " the Python program FluidDyn "
                + fluiddyn.__version__
                + " and FluidSim "
                + fluidsim.get_local_version()
                + ".\n\nIt should not be modified "
                "(except for adding xml comments)."
            )
            path_run = Path(self.path_run)
            info_solver_xml_path = path_run / "info_solver.xml"
            params_xml_path = path_run / "params_simul.xml"

            # save info on the run
            if replace:
                os.remove(info_solver_xml_path)
                os.remove(params_xml_path)

            self.sim.info.solver._save_as_xml(
                path_file=info_solver_xml_path, comment=comment
            )

            self.sim.params._save_as_xml(
                path_file=params_xml_path, comment=comment
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

        dict_classes = self.sim.info.solver.classes.Output.import_classes()

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
        self.print_stdout(
            f"Computation completed in {total_time:8.6g} s\n"
            "path_run =\n" + self.path_run
        )
        if self._has_to_save:
            if hasattr(self.sim, "forcing"):
                self.sim.forcing.compute()
            self.one_time_step()
            if self.sim.output.phys_fields.t_last_save < self.sim.time_stepping.t:
                self.phys_fields.save()

        self.close_files()

        if not self.path_run.startswith(FLUIDSIM_PATH):
            path_base = FLUIDSIM_PATH
            if len(self.params.sub_directory) > 0:
                path_base = os.path.join(path_base, self.params.sub_directory)

            new_path_run = os.path.join(path_base, self.sim.name_run)

            if mpi.rank == 0:
                if not os.path.exists(path_base):
                    os.makedirs(path_base)

                shutil.move(self.path_run, path_base)
                print("move result directory in directory:\n" + new_path_run)

            self.path_run = new_path_run
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
    def init_with_oper_and_state(self):

        oper = self.oper
        self.sum_wavenumbers = oper.sum_wavenumbers
        super().init_with_oper_and_state()

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
        dict_arrays_1time=None,
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
            self._init_files(dict_arrays_1time)

    def _init_path_files(self):
        if hasattr(self, "_name_file"):
            self.path_file = os.path.join(self.output.path_run, self._name_file)

    def _init_files(self, dict_arrays_1time=None):
        if dict_arrays_1time is None:
            dict_arrays_1time = {}
        dict_results = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                self._create_file_from_dict_arrays(
                    self.path_file, dict_results, dict_arrays_1time
                )
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, "r") as f:
                    dset_times = f["times"]
                    self.nb_saved_times = dset_times.shape[0] + 1
                self._add_dict_arrays_to_file(self.path_file, dict_results)
        self.t_last_save = self.sim.time_stepping.t

    def _online_save(self):
        """Save the values at one time. """
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
        self, path_file, dict_matrix, dict_arrays_1time
    ):
        if os.path.exists(path_file):
            print("file NOT created since it already exists!")
        elif mpi.rank == 0:
            with h5py.File(path_file, "w") as f:
                f.attrs["date saving"] = str(datetime.datetime.now()).encode()
                f.attrs["name_solver"] = self.output.name_solver
                f.attrs["name_run"] = self.output.name_run

                self.sim.info._save_as_hdf5(hdf5_parent=f)

                times = np.array([self.sim.time_stepping.t], dtype=np.float64)
                f.create_dataset("times", data=times, maxshape=(None,))

                for k, v in list(dict_arrays_1time.items()):
                    f.create_dataset(k, data=v)

                for k, v in list(dict_matrix.items()):
                    if isinstance(v, numbers.Number):
                        arr = np.array([v], dtype=v.__class__)
                        arr.resize((1,))
                        f.create_dataset(k, data=arr, maxshape=(None,))
                    else:
                        arr = np.array(v)
                        arr.resize((1,) + v.shape)
                        f.create_dataset(
                            k, data=arr, maxshape=((None,) + v.shape)
                        )

    def _add_dict_arrays_to_file(self, path_file, dict_matrix):
        if not os.path.exists(path_file):
            raise ValueError("can not add dict arrays in nonexisting file!")

        elif mpi.rank == 0:
            with h5py.File(path_file, "r+") as f:
                dset_times = f["times"]
                nb_saved_times = dset_times.shape[0]
                dset_times.resize((nb_saved_times + 1,))
                dset_times[nb_saved_times] = self.sim.time_stepping.t
                for k, v in list(dict_matrix.items()):
                    if isinstance(v, numbers.Number):
                        dset_k = f[k]
                        dset_k.resize((nb_saved_times + 1,))
                        dset_k[nb_saved_times] = v
                    else:
                        dset_k = f[k]
                        dset_k.resize((nb_saved_times + 1,) + v.shape)
                        dset_k[nb_saved_times] = v

    def _add_dict_arrays_to_open_file(self, f, dict_arrays, nb_saved_times):
        if mpi.rank == 0:
            dset_times = f["times"]
            dset_times.resize((nb_saved_times + 1,))
            dset_times[nb_saved_times] = self.sim.time_stepping.t
            for k, v in list(dict_arrays.items()):
                dset_k = f[k]
                dset_k.resize((nb_saved_times + 1, v.size))
                dset_k[nb_saved_times] = v

    def _has_to_online_save(self):
        return (
            self.sim.time_stepping.t - self.t_last_save
            >= self.period_save - 1e-14
        )
