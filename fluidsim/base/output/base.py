"""Base module for the output (:mod:`fluidsim.base.output.base`)
======================================================================

.. currentmodule:: fluidsim.base.output.base

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

from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import datetime
import os
import shutil
import numpy as np
import numbers

import fluiddyn

from fluiddyn.util import mpi

from fluiddyn.io import FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH

from fluiddyn.util.util import time_as_str, print_memory_usage

from fluidsim.util.util import load_params_simul


class OutputBase(object):
    """Handle the output."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""
        info_solver.classes.Output._set_child('classes')
        classes = info_solver.classes.Output.classes

        classes._set_child(
            'PrintStdOut',
            attribs={'module_name': 'fluidsim.base.output.print_stdout',
                     'class_name': 'PrintStdOutBase'})

        classes._set_child(
            'PhysFields',
            attribs={'module_name': 'fluidsim.base.output.phys_fields',
                     'class_name': 'PhysFieldsBase'})

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        attribs = {'period_refresh_plots': 1,
                   'ONLINE_PLOT_OK': True,
                   'HAS_TO_SAVE': True,
                   'sub_directory': ''}
        params._set_child('output', attribs=attribs)

        params.output._set_child('periods_save')
        params.output._set_child('periods_print')
        params.output._set_child('periods_plot')

        dict_classes = info_solver.classes.Output.import_classes()
        for Class in dict_classes.values():
            if hasattr(Class, '_complete_params_with_default'):
                try:
                    Class._complete_params_with_default(params)
                except TypeError:
                    try:
                        Class._complete_params_with_default(
                            params, info_solver)
                    except TypeError as e:
                        e.args += ('for class: ' + repr(Class),)
                        raise

    def __init__(self, sim):
        params = sim.params
        self.sim = sim
        self.params = params.output

        self.has_to_save = self.params.HAS_TO_SAVE
        self.name_solver = sim.info.solver.short_name

        # initialisation name_run and path_run
        list_for_name_run = self.create_list_for_name_run()
        list_for_name_run.append(time_as_str())
        self.name_run = '_'.join(list_for_name_run)

        self.sim.name_run = self.name_run

        if not params.NEW_DIR_RESULTS:
            try:
                self.path_run = params.path_run
            except AttributeError:
                params.NEW_DIR_RESULTS = True
                print('Strange: params.NEW_DIR_RESULTS == False '
                      'but no params.path_run')

            # if has_to_save, we verify the correspondence between the
            # resolution of the simulation and the resolution of the
            # previous simulation saved in this directory
            if self.has_to_save:
                if mpi.rank == 0:
                    try:
                        params_dir = load_params_simul(path_dir=self.path_run)
                    except:
                        raise ValueError(
                            'Strange, no info_simul.h5 in self.path_run')

                    if (params.oper.nx != params_dir.oper.nx or
                            params.oper.ny != params_dir.oper.ny):
                        params.NEW_DIR_RESULTS = True
                        print("""
Warning: params.NEW_DIR_RESULTS is False but the resolutions of the simulation
         and of the simulation in the directory self.path_run are different
         we put params.NEW_DIR_RESULTS = True""")
                if mpi.nb_proc > 1:
                    params.NEW_DIR_RESULTS = \
                        mpi.comm.bcast(params.NEW_DIR_RESULTS)

        if params.NEW_DIR_RESULTS:

            if FLUIDDYN_PATH_SCRATCH is not None:
                path_base = FLUIDDYN_PATH_SCRATCH
            else:
                path_base = FLUIDSIM_PATH

            if len(params.output.sub_directory) > 0:
                path_base = os.path.join(
                    path_base, params.output.sub_directory)

            self.path_run = os.path.join(path_base, self.sim.name_run)

            if mpi.rank == 0:
                params._set_attrib('path_run', self.path_run)
                if not os.path.exists(self.path_run):
                    os.makedirs(self.path_run)

        dico_classes = sim.info.solver.classes.Output.import_classes()

        PrintStdOut = dico_classes['PrintStdOut']
        self.print_stdout = PrintStdOut(self)

        if not self.params.ONLINE_PLOT_OK:
            for k in self.params.periods_plot._attribs:
                self.params.periods_plot[k] = 0.

        if not self.has_to_save:
            for k in self.params.periods_save._attribs:
                self.params.periods_save[k] = 0.

    def create_list_for_name_run(self):
        list_for_name_run = [self.name_solver]
        if len(self.sim.params.short_name_type_run) > 0:
            list_for_name_run.append(self.sim.params.short_name_type_run)
        list_for_name_run.append(self.sim.oper.produce_str_describing_oper())

        return list_for_name_run

    def init_with_oper_and_state(self):
        sim = self.sim

        self.oper = sim.oper

        if mpi.rank == 0:
            # print info on the run
            specifications = (
                ', ' + sim.params.time_stepping.type_time_scheme + ' and ')
            if mpi.nb_proc == 1:
                specifications += 'sequential,\n'
            else:
                specifications += 'parallel ({} proc.)\n'.format(mpi.nb_proc)
            self.print_stdout(
                '\nsolver ' + self.name_solver + specifications +
                self.sim.oper.produce_long_str_describing_oper() +
                'path_run =\n' + self.path_run + '\n' +
                'init_fields.type: ' + sim.params.init_fields.type)

        self.save_info_solver_params_xml()
        
        if mpi.rank == 0:
            plt.ion()

        if self.sim.state.is_initialized:
            self.init_with_initialized_state()
    
    def save_info_solver_params_xml(self, replace=False):
        comment = ('This file has been created by'
                   ' the Python program FluidDyn ' + fluiddyn.__version__ +
                   '.\n\nIt should not be modified '
                   '(except for adding xml comments).')
        info_solver_xml_path = self.path_run + '/info_solver.xml'
        params_xml_path = self.path_run + '/params_simul.xml'
        
        if mpi.rank == 0 and self.has_to_save and self.sim.params.NEW_DIR_RESULTS:
            # save info on the run
            if replace:
                os.remove(info_solver_xml_path)
                os.remove(params_xml_path)
                
            self.sim.info.solver._save_as_xml(
                path_file=info_solver_xml_path,
                comment=comment)

            self.sim.params._save_as_xml(
                path_file=params_xml_path,
                comment=comment)        

    def init_with_initialized_state(self):

        if (hasattr(self, 'has_been_initialized_with_state') and
                self.has_been_initialized_with_state):
            return
        else:
            self.has_been_initialized_with_state = True

        self.print_stdout('Initialization outputs:')

        self.print_stdout.complete_init_with_state()

        dico_classes = self.sim.info.solver.classes.Output.import_classes()

        # The class PrintStdOut has already been instantiated.
        dico_classes.pop('PrintStdOut')

        for Class in dico_classes.values():
            if mpi.rank == 0:
                print(Class, Class._tag)
            self.__dict__[Class._tag] = Class(self)

        print_memory_usage(
            '\nMemory usage at the end of init. (equiv. seq.)')

        try:
            self.print_size_in_Mo(self.sim.state.state_fft, 'state_fft')
        except AttributeError:
            self.print_size_in_Mo(self.sim.state.state_phys, 'state_phys')

    def one_time_step(self):

        for k in self.params.periods_print._attribs:
            period = self.params.periods_print.__dict__[k]
            if period != 0:
                self.__dict__[k].online_print()

        if self.params.ONLINE_PLOT_OK:
            for k in self.params.periods_plot._attribs:
                period = self.params.periods_plot.__dict__[k]
                if period != 0:
                    self.__dict__[k].online_plot()

        if self.has_to_save:
            for k in self.params.periods_save._attribs:
                period = self.params.periods_save.__dict__[k]
                if period != 0:
                    self.__dict__[k].online_save()

    def figure_axe(self, numfig=None, size_axe=None):
        if mpi.rank == 0:
            if size_axe is None:
                x_left_axe = 0.12
                z_bottom_axe = 0.1
                width_axe = 0.85
                height_axe = 0.84
                size_axe = [x_left_axe, z_bottom_axe,
                            width_axe, height_axe]
            if numfig is None:
                fig = plt.figure()
            else:
                fig = plt.figure(numfig)
                fig.clf()
            axe = fig.add_axes(size_axe)
            return fig, axe

    def end_of_simul(self, total_time):
        self.print_stdout(
            'Computation completed in {0:8.6g} s\n'.format(total_time) +
            'path_run =\n'+self.path_run)
        if self.has_to_save:
            self.phys_fields.save()
        if mpi.rank == 0 and self.has_to_save:
            self.print_stdout.close()

            for k in self.params.periods_save._attribs:
                period = self.params.periods_save.__dict__[k]
                if period != 0:
                    if hasattr(self.__dict__[k], 'close_file'):
                        self.__dict__[k].close_file()

        if (not self.path_run.startswith(FLUIDSIM_PATH) and mpi.rank == 0):
            path_base = FLUIDSIM_PATH
            if len(self.params.sub_directory) > 0:
                path_base = os.path.join(path_base, self.params.sub_directory)

            if not os.path.exists(path_base):
                os.makedirs(path_base)

            new_path_run = os.path.join(path_base, self.sim.name_run)
            shutil.move(self.path_run, path_base)
            print('move result directory in directory:\n' + new_path_run)
            self.path_run = new_path_run
            for spec_output in self.__dict__.values():
                if isinstance(spec_output, SpecificOutput):
                    try:
                        spec_output.init_path_files()
                    except AttributeError:
                        pass

    def compute_energy(self):
        return 0.

    def print_size_in_Mo(self, arr, string=None):
        if string is None:
            string = 'Size of ndarray (equiv. seq.)'
        else:
            string = 'Size of '+string+' (equiv. seq.)'
        mem = arr.nbytes*1.e-6
        if mpi.nb_proc > 1:
            mem = mpi.comm.allreduce(mem, op=mpi.MPI.SUM)
        self.print_stdout(string.ljust(30)+': {0} Mo'.format(mem))


class OutputBasePseudoSpectral(OutputBase):

    def init_with_oper_and_state(self):

        oper = self.sim.oper
        self.sum_wavenumbers = oper.sum_wavenumbers
        # self.fft2 = oper.fft2
        # self.ifft2 = oper.ifft2
        # # really necessary here?
        # self.vecfft_from_rotfft = oper.vecfft_from_rotfft
        # self.rotfft_from_vecfft = oper.rotfft_from_vecfft

        super(OutputBasePseudoSpectral, self).init_with_oper_and_state()


class SpecificOutput(object):
    """Small class for features useful for specific outputs"""

    def __init__(self, output, period_save=0, period_plot=0,
                 has_to_plot_saved=False,
                 dico_arrays_1time=None):

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
        self.t_last_show = 0.

        self.init_path_files()

        if self.has_to_plot and mpi.rank == 0:
            self.init_online_plot()

        if not output.has_to_save:
            self.period_save = 0.

        if self.period_save != 0.:
            self.init_files(dico_arrays_1time)

    def init_path_files(self):
        if hasattr(self, '_name_file'):
            self.path_file = os.path.join(
                self.output.path_run, self._name_file)

    def init_files(self, dico_arrays_1time=None):
        if dico_arrays_1time is None:
            dico_arrays_1time = {}
        dico_results = self.compute()
        if mpi.rank == 0:
            if not os.path.exists(self.path_file):
                self.create_file_from_dico_arrays(
                    self.path_file, dico_results, dico_arrays_1time)
                self.nb_saved_times = 1
            else:
                with h5py.File(self.path_file, 'r') as f:
                    dset_times = f['times']
                    self.nb_saved_times = dset_times.shape[0]+1
                self.add_dico_arrays_to_file(self.path_file,
                                             dico_results)
        self.t_last_save = self.sim.time_stepping.t

    def online_save(self):
        """Save the values at one time. """
        tsim = self.sim.time_stepping.t
        if (tsim - self.t_last_save >= self.period_save):
            self.t_last_save = tsim
            dico_results = self.compute()
            if mpi.rank == 0:
                self.add_dico_arrays_to_file(self.path_file,
                                             dico_results)
                self.nb_saved_times += 1
                if self.has_to_plot:
                    self._online_plot(dico_results)
                    if (tsim - self.t_last_show >= self.period_show):
                        self.t_last_show = tsim
                        self.fig.canvas.draw()

    def create_file_from_dico_arrays_old(self, path_file,
                                         dico_arrays, dico_arrays_1time):
        if os.path.exists(path_file):
            print('file NOT created since it already exists!')
        elif mpi.rank == 0:
            with h5py.File(path_file, 'w') as f:
                f.attrs['date saving'] = str(datetime.datetime.now())
                f.attrs['name_solver'] = self.output.name_solver
                f.attrs['name_run'] = self.output.name_run

                self.sim.info._save_as_hdf5(hdf5_parent=f)

                times = np.array([self.sim.time_stepping.t])
                f.create_dataset(
                    'times', data=times, maxshape=(None,))

                for k, v in dico_arrays_1time.iteritems():
                    f.create_dataset(k, data=v)

                for k, v in dico_arrays.iteritems():
                    v.resize([1, v.size])
                    f.create_dataset(
                        k, data=v, maxshape=(None, v.size))

    def create_file_from_dico_arrays(self, path_file,
                                     dico_matrix, dico_arrays_1time):
        if os.path.exists(path_file):
            print('file NOT created since it already exists!')
        elif mpi.rank == 0:
            with h5py.File(path_file, 'w') as f:
                f.attrs['date saving'] = str(datetime.datetime.now())
                f.attrs['name_solver'] = self.output.name_solver
                f.attrs['name_run'] = self.output.name_run

                self.sim.info._save_as_hdf5(hdf5_parent=f)

                times = np.array([self.sim.time_stepping.t], dtype=np.float64)
                f.create_dataset(
                    'times', data=times, maxshape=(None,))

                for k, v in dico_arrays_1time.iteritems():
                    f.create_dataset(k, data=v)

                for k, v in dico_matrix.iteritems():
                    if isinstance(v, numbers.Number):
                        arr = np.array([v], dtype=v.__class__)
                        arr.resize((1,))
                        f.create_dataset(
                            k, data=arr, maxshape=(None,))
                    else:
                        arr = np.array(v)
                        arr.resize((1,) + v.shape)
                        f.create_dataset(
                            k, data=arr, maxshape=((None,) + v.shape))

    def add_dico_arrays_to_file_old(self, path_file, dico_arrays):
        if not os.path.exists(path_file):
            raise ValueError('can not add dico arrays in nonexisting file!')
        elif mpi.rank == 0:
            with h5py.File(path_file, 'r+') as f:
                dset_times = f['times']
                nb_saved_times = dset_times.shape[0]
                dset_times.resize((nb_saved_times+1,))
                dset_times[nb_saved_times] = self.sim.time_stepping.t
                for k, v in dico_arrays.iteritems():
                    dset_k = f[k]
                    dset_k.resize((nb_saved_times+1, v.size))
                    dset_k[nb_saved_times] = v

    def add_dico_arrays_to_file(self, path_file, dico_matrix):
        if not os.path.exists(path_file):
            raise ValueError('can not add dico matrix in nonexisting file!')
        elif mpi.rank == 0:
            with h5py.File(path_file, 'r+') as f:
                dset_times = f['times']
                nb_saved_times = dset_times.shape[0]
                dset_times.resize((nb_saved_times+1,))
                dset_times[nb_saved_times] = self.sim.time_stepping.t
                for k, v in dico_matrix.iteritems():
                    if isinstance(v, numbers.Number):
                        dset_k = f[k]
                        dset_k.resize((nb_saved_times+1,))
                        dset_k[nb_saved_times] = v
                    else:
                        dset_k = f[k]
                        dset_k.resize((nb_saved_times+1,) + v.shape)
                        dset_k[nb_saved_times] = v

    def add_dico_arrays_to_open_file(self, f, dico_arrays, nb_saved_times):
        if mpi.rank == 0:
            dset_times = f['times']
            dset_times.resize((nb_saved_times+1,))
            dset_times[nb_saved_times] = self.sim.time_stepping.t
            for k, v in dico_arrays.iteritems():
                dset_k = f[k]
                dset_k.resize((nb_saved_times+1, v.size))
                dset_k[nb_saved_times] = v

    def _has_to_online_save(self):
        return (self.sim.time_stepping.t - self.t_last_save >=
                self.period_save - 1e-14)
