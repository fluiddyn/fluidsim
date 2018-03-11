"""Physical fields output (:mod:`fluidsim.base.output.phys_fields`)
===================================================================

Provides:

.. autoclass:: PhysFieldsBase
   :members:
   :private-members:

.. autoclass:: MoviesBasePhysFields1D
   :members:
   :private-members:

.. autoclass:: PhysFieldsBase1D
   :members:
   :private-members:

.. autoclass:: MoviesBasePhysFields2D
   :members:
   :private-members:

.. autoclass:: PhysFieldsBase2D
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function

from builtins import str, map
from past.builtins import basestring

import re
import os
import datetime
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import h5py
import h5netcdf

from fluiddyn.util import mpi
from .base import SpecificOutput
from .movies import MoviesBase1D, MoviesBase2D
from ..params import Parameters

cfg_h5py = h5py.h5.get_config()

if cfg_h5py.mpi:
    ext = 'h5'
    h5pack = h5py
else:
    ext = 'nc'
    h5pack = h5netcdf


def _create_variable(group, key, field):
    if ext == 'nc':
        if field.ndim == 0:
            dimensions = tuple()
        elif field.ndim == 1:
            dimensions = ('x',)
        elif field.ndim == 2:
            dimensions = ('y', 'x')
        elif field.ndim == 3:
            dimensions = ('z', 'y', 'x')
        try:
            group.create_variable(key, data=field, dimensions=dimensions)
        except AttributeError:
            raise ValueError(
                'Error while creating a netCDF4 variable using group'
                ' of type {} for key {}'.format(
                    type(group), key))
    else:
        try:
            group.create_dataset(key, data=field)
        except AttributeError:
            raise ValueError(
                'Error while creating a HDF5 dataset using group'
                ' of type {} for key {}'.format(
                    type(group), key))


class PhysFieldsBase(SpecificOutput):
    """Manage the output of physical fields."""

    _tag = 'phys_fields'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'phys_fields'
        params.output._set_child(tag,
                                 attribs={'field_to_plot': 'ux',
                                          'file_with_it': False})

        params.output.periods_save._set_attrib(tag, 0)
        params.output.periods_plot._set_attrib(tag, 0)

    def __init__(self, output):
        params = output.sim.params

        super(PhysFieldsBase, self).__init__(
            output,
            period_save=params.output.periods_save.phys_fields,
            period_plot=params.output.periods_plot.phys_fields)

        self.field_to_plot = params.output.phys_fields.field_to_plot

        if self.period_save == 0 and self.period_plot == 0:
            return

        self.t_last_save = self.sim.time_stepping.t
        self.t_last_plot = self.sim.time_stepping.t

    def _init_files(self, dico_arrays_1time=None):
        pass

    def _init_online_plot(self):
        pass

    def _online_save(self):
        """Online save."""
        tsim = self.sim.time_stepping.t
        if self._has_to_online_save():
            self.t_last_save = tsim
            self.save()

    def _online_plot(self):
        """Online plot."""
        tsim = self.sim.time_stepping.t
        if (tsim - self.t_last_plot >= self.period_plot):
            self.t_last_plot = tsim
            itsim = self.sim.time_stepping.it
            self.plot(numfig=itsim,
                      key_field=self.params.output.phys_fields.field_to_plot)

    def save(self, state_phys=None, params=None, particular_attr=None):
        if state_phys is None:
            state_phys = self.sim.state.state_phys
        if params is None:
            params = self.params

        time = self.sim.time_stepping.t

        path_run = self.output.path_run

        if mpi.rank == 0 and not os.path.exists(path_run):
            os.mkdir(path_run)

        if (self.period_save < 0.001 or
                self.params.output.phys_fields.file_with_it):
            name_save = 'state_phys_t{:07.3f}_it={}.{}'.format(
                time, self.sim.time_stepping.it, ext)
        else:
            name_save = 'state_phys_t{:07.3f}.{}'.format(time, ext)

        path_file = os.path.join(path_run, name_save)
        if os.path.exists(path_file):
            name_save = 'state_phys_t{:07.3f}_it={}.{}'.format(
                time, self.sim.time_stepping.it, ext)
            path_file = os.path.join(path_run, name_save)
        to_print = 'save state_phys in file ' + name_save
        self.output.print_stdout(to_print)

        # FIXME: bad condition below when run sequentially, with MPI enabled h5py
        # As a workaround the instantiation is made with h5pack
        if mpi.nb_proc == 1 or not cfg_h5py.mpi:
            if mpi.rank == 0:
                # originally:
                # f = h5netcdf.File(...
                f = h5pack.File(path_file, 'w')
                group_state_phys = f.create_group("state_phys")
                group_state_phys.attrs['what'] = 'obj state_phys for solveq2d'
                group_state_phys.attrs['name_type_variables'] = state_phys.info
                group_state_phys.attrs['time'] = time
                group_state_phys.attrs['it'] = self.sim.time_stepping.it
        else:
            # originally:
            # f = h5py.File(...
            f = h5pack.File(path_file, 'w', driver='mpio', comm=mpi.comm)
            group_state_phys = f.create_group("state_phys")
            group_state_phys.attrs['what'] = 'obj state_phys for solveq2d'
            group_state_phys.attrs['name_type_variables'] = state_phys.info

            group_state_phys.attrs['time'] = time
            group_state_phys.attrs['it'] = self.sim.time_stepping.it

        if mpi.nb_proc == 1:
            for k in state_phys.keys:
                field_seq = state_phys.get_var(k)
                _create_variable(group_state_phys, k, field_seq)
        elif not cfg_h5py.mpi:
            for k in state_phys.keys:
                field_loc = state_phys.get_var(k)
                field_seq = self.oper.gather_Xspace(field_loc)
                if mpi.rank == 0:
                    _create_variable(group_state_phys, k, field_seq)
        else:
            for k in state_phys.keys:
                field_loc = state_phys.get_var(k)
                dset = group_state_phys.create_dataset(
                    k, self.oper.shapeX_seq, dtype=field_loc.dtype)
                f.atomic = False
                xstart = self.oper.seq_index_firstK0
                xend = self.oper.seq_index_firstK0 + self.oper.shapeX_loc[0]
                ystart = self.oper.seq_index_firstK1
                yend = self.oper.seq_index_firstK1 + self.oper.shapeX_loc[1]
                with dset.collective:
                    dset[xstart:xend, ystart:yend, :] = field_loc
            f.close()
            if mpi.rank == 0:
                f = h5pack.File(path_file, 'w')

        if mpi.rank == 0:
            f.attrs['date saving'] = str(datetime.datetime.now()).encode()
            f.attrs['name_solver'] = self.output.name_solver
            f.attrs['name_run'] = self.output.name_run
            if particular_attr is not None:
                f.attrs['particular_attr'] = particular_attr

            self.sim.info._save_as_hdf5(hdf5_parent=f)
            gp_info = f['info_simul']
            gf_params = gp_info['params']
            gf_params.attrs['SAVE'] = 1
            gf_params.attrs['NEW_DIR_RESULTS'] = 1
            f.close()

    def _select_field(self, field=None, key_field=None):
        keys_state_phys = self.sim.info.solver.classes.State['keys_state_phys']
        keys_computable = self.sim.info.solver.classes.State['keys_computable']

        if field is None:
            if key_field is None:
                field_to_plot = self.params.output.phys_fields.field_to_plot
                if (field_to_plot in keys_state_phys or
                        field_to_plot in keys_computable):
                    key_field = field_to_plot
                else:
                    if 'q' in keys_state_phys:
                        key_field = 'q'
                    elif 'rot' in keys_state_phys:
                        key_field = 'rot'
                    else:
                        key_field = keys_state_phys[0]
                field_loc = self.sim.state.get_var(key_field)
            else:
                field_loc = self.sim.state.get_var(key_field)
        else:
            key_field = 'given field'
            field_loc = field

        if mpi.nb_proc > 1:
            field = self.oper.gather_Xspace(field_loc)
        else:
            field = field_loc

        return field, key_field


class MoviesBasePhysFields1D(MoviesBase1D):
    def _ani_init(self, key_field, numfig, dt_equations, tmin, tmax,
                  fig_kw, **kwargs):
        """Initialize list of files and times, pcolor plot, quiver and colorbar.
        """
        self._set_path()
        self._ani_pathfiles = sorted(glob(os.path.join(
            self.path, 'state_phys*.[hn]*')))
        self._ani_t_actual = np.array(list(
            map(time_from_path, self._ani_pathfiles)))

        if dt_equations is None:
            dt_equations = np.median(np.diff(self._ani_t_actual))
            print('dt_equations = {:.4f}'.format(dt_equations))

        if tmax is None:
            tmax = self._ani_t_actual.max()

        super(MoviesBasePhysFields1D, self)._ani_init(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs)


    def _ani_get_field(self, time, key=None, need_uxuy=True):
        """Get field from saved physical fields."""
        if key is None:
            key = self._ani_key
        idx, t_actual = self._ani_get_t_actual(time)

        with h5py.File(self._ani_pathfiles[idx]) as f:
            field = f['state_phys'][key].value

        return field


class PhysFieldsBase1D(PhysFieldsBase, MoviesBasePhysFields1D):
    def plot(self, key_field=None, field=None, numfig=None):
        field, key_field = self._select_field(field, key_field)

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe(size_axe=None)
            else:
                fig, ax = self.output.figure_axe(numfig=numfig,
                                                 size_axe=None)
            xs = self.oper.xs

            ax.plot(xs, field)


def time_from_path(path):
    '''Regular expression search to extract time from filename.'''
    filename = os.path.basename(path)
    pattern = r'''
        (?!t)     # text after t but exclude it
        [0-9]+    # a couple of digits
        \.        # the decimal point
        [0-9]+    # a couple of digits
    '''
    match = re.search(pattern, filename, re.VERBOSE)
    t = float(match.group(0))
    return t


class MoviesBasePhysFields2D(MoviesBase2D):
    """Methods required to animate physical fields HDF5 files."""

    def _ani_init(self, key_field, numfig, dt_equations, tmin, tmax,
                  fig_kw, **kwargs):
        """Initialize list of files and times, pcolor plot, quiver and colorbar.
        """
        self._set_path()
        self._ani_pathfiles = sorted(glob(os.path.join(
            self.path, 'state_phys*.[hn]*')))
        self._ani_t_actual = np.array(list(
            map(time_from_path, self._ani_pathfiles)))

        if dt_equations is None:
            dt_equations = np.median(np.diff(self._ani_t_actual))
            print('dt_equations = {:.4f}'.format(dt_equations))

        if tmax is None:
            tmax = self._ani_t_actual.max()

        super(MoviesBasePhysFields2D, self)._ani_init(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs)

        dt_file = (self._ani_t_actual[-1] - self._ani_t_actual[0]) / (
            self._ani_t_actual.size)
        if dt_equations < dt_file / 4:
            raise ValueError('dt_equations < dt_file / 4')

        self._has_uxuy = self.sim.state.has_vars('ux', 'uy')
        field, ux, uy = self._ani_get_field(0)

        INSET = True if 'INSET' not in kwargs else kwargs['INSET']
        self._ani_init_fig(field, ux, uy, INSET, **kwargs)
        self._ani_clim = kwargs.get('clim')
        self._ani_set_clim()

    def _ani_init_fig(self, field, ux=None, uy=None, INSET=True, **kwargs):
        """Initialize only the figure and related matplotlib objects. This
        method is shared by both ``animate`` and ``online_plot``
        functionalities.

        """
        self._step = step = 1 if 'step' not in kwargs else kwargs['step']
        self._QUIVER = True if 'QUIVER' not in kwargs else kwargs['QUIVER']

        x, y = self._select_axis(shape=field.shape)
        XX, YY = np.meshgrid(x[::step], y[::step])
        field = field[::step, ::step]

        self._ani_im = self._ani_ax.pcolormesh(XX, YY, field)
        self._ani_cbar = self._ani_fig.colorbar(self._ani_im)

        self._has_uxuy = self.sim.state.has_vars('ux', 'uy')

        if self._has_uxuy and self._QUIVER:
            self._ani_quiver, vmax = self._quiver_plot(
                self._ani_ax, ux, uy, XX, YY)

        if not self.sim.time_stepping.is_simul_completed():
            INSET = False

        try:
            self.output.spatial_means
        except AttributeError:
            INSET = False

        self._ANI_INSET = INSET
        if self._ANI_INSET:
            try:
                self._ani_spatial_means_t, self._ani_spatial_means_key = (
                    self._get_spatial_means())
            except FileNotFoundError:
                print('No spatial means file => no inset plot.')
                self._ANI_INSET = False
                return

            left, bottom, width, height = [0.53, 0.67, 0.2, 0.2]
            ax2 = self._ani_fig.add_axes([left, bottom, width, height])

            ax2.set_xlabel('t', labelpad=0.1)
            ax2.set_ylabel('E', labelpad=0.1)

            # Format of the ticks in ylabel
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

            ax2.set_xlim(0, self._ani_spatial_means_t.max())
            # Correct visualization inset_animation 10% of the difference
            # value_max-value-min
            ax2.set_ylim(
                self._ani_spatial_means_key.min(),
                self._ani_spatial_means_key.max() + (
                    0.1 * abs(self._ani_spatial_means_key.min() -
                              self._ani_spatial_means_key.max())))

            ax2.plot(
                self._ani_spatial_means_t, self._ani_spatial_means_key,
                linewidth=0.8, color='grey', alpha=0.4)
            self._ani_im_inset = ax2.plot([0], [0], color='red')

    def _quiver_plot(self, ax, vecx='ux', vecy='uy', XX=None, YY=None):
        '''Make a quiver plot on axis `ax`.'''
        pass

    def _select_axis(self, xlabel='x', ylabel='y', shape=None):
        '''Get 1D arrays for setting the axes.'''

        x, y = super(MoviesBasePhysFields2D, self)._select_axis(xlabel, ylabel)
        if shape is not None and (y.shape[0], x.shape[0]) != shape:
            path_file = os.path.join(self.path, 'params_simul.xml')
            params = Parameters(path_file=path_file)
            x = np.arange(0, params.oper.Lx, params.oper.nx)
            y = np.arange(0, params.oper.Ly, params.oper.ny)

        return x, y

    def _ani_get_field(self, time, key=None, need_uxuy=True):
        """Get field, ux, uy from saved physical fields."""

        if key is None:
            key = self._ani_key
        idx, t_actual = self._ani_get_t_actual(time)

        with h5py.File(self._ani_pathfiles[idx]) as f:
            field = f['state_phys'][key].value

            if need_uxuy and self._has_uxuy:
                try:
                    ux = f['state_phys']['ux'].value
                    uy = f['state_phys']['uy'].value
                except KeyError:
                    ux = f['state_phys']['vx'].value
                    uy = f['state_phys']['vy'].value

        if need_uxuy:
            if self._has_uxuy:
                return field, ux, uy
            else:
                return field, None, None
        else:
            return field

    def _ani_update(self, frame, **fargs):
        """Loads data and updates figure."""
        time = self._ani_t[frame]
        step = self._step

        field, ux, uy = self._ani_get_weighted_field(time)
        field = field[::step, ::step]
        # tricky: https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
        field = field[:-1, :-1]
        
        # Update figure, quiver and colorbar
        self._ani_im.set_array(field.flatten())
        
        if self._has_uxuy and self._QUIVER:
            vmax = np.max(np.sqrt(ux ** 2 + uy ** 2))
            self._ani_quiver.set_UVC(ux[::self._skip, ::self._skip]/vmax,
                                     uy[::self._skip, ::self._skip]/vmax)
        else:
            vmax = None

        self._ani_im.autoscale()
        self._ani_set_clim()

        # INLET ANIMATION
        if self._ANI_INSET:
            idx_spatial = np.abs(self._ani_spatial_means_t - time).argmin()
            t = self._ani_spatial_means_t
            E = self._ani_spatial_means_key

            self._ani_im_inset[0].set_data(
                t[:idx_spatial], E[:idx_spatial])

        self._set_title(self._ani_ax, self._ani_key, time, vmax)

    def _set_title(self, ax, key, time, vmax=None):
        title = (key +
                 ', $t = {0:.3f}$, '.format(time) +
                 self.output.name_solver +
                 ', $n_x = {0:d}$'.format(self.params.oper.nx))
        if vmax is not None:
            title += r', $|\vec{v}|_{max} = $' + '{0:.3f}'.format(vmax)
        ax.set_title(title)

    def _ani_set_clim(self):
        """Maintains a constant colorbar throughout the animation."""

        clim = self._ani_clim
        if clim is not None:
            self._ani_im.set_clim(*clim)
            self._ani_cbar.set_clim(*clim)
            ticks = np.linspace(*clim, num=21, endpoint=True)
            self._ani_cbar.set_ticks(ticks)

    def _ani_get_weighted_field(self, time):
        """Get weighted field between two saved files."""
        idx, t_actual = self._ani_get_t_actual(time)
        # If time > time last frame --> frame == last frame.
        if idx + 1 >= len(self._ani_t_actual) and time > t_actual:
            t = self._ani_t_actual[idx - 1]
            field, ux, uy = self._ani_get_field(t)

        elif t_actual != time:
            if t_actual < time:
                idx_0 = idx
                idx_1 = idx + 1
            elif t_actual > time:
                idx_0 = idx - 1
                idx_1 = idx

            dt_save = self._ani_t_actual[idx_1] - self._ani_t_actual[idx_0]
            weight_0 = 1 - np.abs(
                time - self._ani_t_actual[idx_0]) / dt_save
            weight_1 = 1 - np.abs(
                time - self._ani_t_actual[idx_1]) / dt_save

            t0 = self._ani_t_actual[idx_0]
            field_0, ux_0, uy_0 = self._ani_get_field(t0)

            t1 = self._ani_t_actual[idx_1]
            field_1, ux_1, uy_1 = self._ani_get_field(t1)

            field = field_0 * weight_0 + field_1 * weight_1
            if self._has_uxuy:
                ux = ux_0 * weight_0 + ux_1 * weight_1
                uy = uy_0 * weight_0 + uy_1 * weight_1
            else:
                ux = uy = None
        else:
            t = self._ani_t_actual[idx]
            field, ux, uy = self._ani_get_field(t)

        return field, ux, uy

    def _get_spatial_means(self, key_spatial='E'):
        """ Get field for the inset plot."""
        # Check if key_spatial can be loaded.
        keys_spatial = ['E', 'EK', 'EA']
        if key_spatial not in keys_spatial:
            raise ValueError('key_spatial not in spatial means keys.')
        # Load data for inset plot
        dico = self.output.spatial_means.load()
        t = dico['t']
        E = dico[key_spatial]

        return t, E


class PhysFieldsBase2D(PhysFieldsBase, MoviesBasePhysFields2D):

    def _init_online_plot(self):
        self._ani_key = self.params.output.phys_fields.field_to_plot
        self._has_uxuy = self.sim.state.has_vars('ux', 'uy')

        field, _ = self._select_field(key_field=self._ani_key)
        if self._has_uxuy:
            ux, _ = self._select_field(key_field='ux')
            uy, _ = self._select_field(key_field='uy')
        else:
            ux = uy = None

        if mpi.rank == 0:
            self._set_font()
            self._ani_fig, self._ani_ax = plt.subplots()
            self._ani_init_fig(field, ux, uy)
            self._ani_im.autoscale()
            self._ani_fig.canvas.draw()
            plt.pause(1e-6)

    def _online_plot(self):
        """Online plot."""
        tsim = self.sim.time_stepping.t
        if (tsim - self.t_last_plot >= self.period_plot):
            self.t_last_plot = tsim
            key_field = self.params.output.phys_fields.field_to_plot
            field, _ = self._select_field(key_field=key_field)
            if self._has_uxuy:
                ux, _ = self._select_field(key_field='ux')
                uy, _ = self._select_field(key_field='uy')

            if mpi.rank == 0:
                field = field[:-1, :-1]

                # Update figure, quiver and colorbar
                self._ani_im.set_array(field.flatten())
                if self._has_uxuy:
                    vmax = np.max(np.sqrt(ux ** 2 + uy ** 2))
                    self._ani_quiver.set_UVC(ux[::self._skip, ::self._skip] / vmax,
                                             uy[::self._skip, ::self._skip] / vmax)
                else:
                    vmax = None

                self._set_title(self._ani_ax, self._ani_key, tsim, vmax)

                self._ani_im.autoscale()
                self._ani_fig.canvas.draw()
                plt.pause(1e-6)

    def plot(self, key_field=None, field=None, time=None,
             QUIVER=True, vecx='ux', vecy='uy',
             nb_contours=20, type_plot='contourf', iz=0, vmin=None, vmax=None,
             cmap='viridis', numfig=None):
        """Plot a field.

        This function is surely buggy! It has to be fixed.

        Parameters
        ----------

        field : {str, np.ndarray}, optional

        time : number, optional

        QUIVER : True

        vecx : 'ux'

        vecy : 'uy'

        nb_contours : 20

        type_plot : 'contourf'

        iz : 0

        vmin : None

        vmax : None

        cmap : 'viridis'

        numfig : None

        """

        is_field_ready = False

        self._has_uxuy = self.sim.state.has_vars('ux', 'uy')

        if field is None and key_field is None:
            key_field = self.field_to_plot
        elif isinstance(field, np.ndarray) and key_field is None:
            key_field = 'given array'
            is_field_ready = True
        elif field is not None and key_field is None:
            key_field, field = field, key_field

        assert key_field is not None

        keys_state_phys = self.sim.state.keys_state_phys
        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

        if time is None and not is_field_ready:
            # we have to get the field from the state
            time = self.sim.time_stepping.t
            field = self._get_field_seq(key_field)
            if QUIVER:
                vecx = self._get_field_seq(vecx)
                vecy = self._get_field_seq(vecy)
        else:
            # we have to get the field from a file
            self._set_path()

            if key_field not in self.sim.state.keys_state_phys:
                raise ValueError('key not in state.keys_state_phys')

            self._ani_pathfiles = sorted(glob(os.path.join(
                self.path, 'state_phys*')))
            self._ani_t_actual = np.array(list(
                map(time_from_path, self._ani_pathfiles)))

            idx, t_actual = self._ani_get_t_actual(time)
            time = t_actual

            if QUIVER:
                field, vecx, vecy = self._ani_get_field(time, key_field)
            else:
                field = self._ani_get_field(time, key_field, need_uxuy=False)

        if field.ndim == 3:
            field = field[iz]
            vecx, vecy = vecx[iz], vecy[iz]

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe()
            else:
                fig, ax = self.output.figure_axe(numfig=numfig)
            x_seq = self.oper.x_seq
            y_seq = self.oper.y_seq
            [XX_seq, YY_seq] = np.meshgrid(x_seq, y_seq)
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError:
                print('Use matplotlib >= 1.5.0 for new standard colorschemes.\
                       Installed matplotlib :' + plt.matplotlib.__version__)
                cmap = plt.get_cmap('jet')

            if type_plot == 'contourf':
                contours = ax.contourf(
                    x_seq, y_seq, field,
                    nb_contours, vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(contours)
                fig.contours = contours
            elif type_plot == 'pcolor':
                pc = ax.pcolormesh(x_seq, y_seq, field,
                                   vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(pc)
        else:
            ax = None

        if QUIVER:
            quiver, vmax = self._quiver_plot(ax, vecx, vecy)
        else:
            vmax = None

        if mpi.rank == 0:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            self._set_title(ax, key_field, time, vmax)

            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(1e-3)

    def _compute_skip_quiver(self):
        # 4% of the Lx it is a great separation between vector arrows.
        delta_quiver = 0.04 * self.oper.Lx
        skip = (self.oper.nx_seq / self.oper.Lx) * delta_quiver
        skip = int(np.round(skip))
        if skip < 1:
            skip = 1
        return skip

    def _get_field_seq(self, key):
        field_loc = self.sim.state.get_var(key)
        if mpi.nb_proc > 1:
            return self.oper.gather_Xspace(field_loc)
        else:
            return field_loc

    def _quiver_plot(self, ax, vecx='ux', vecy='uy', XX=None, YY=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        if isinstance(vecx, basestring):
            vecx = self._get_field_seq(vecx)

        if isinstance(vecy, basestring):
            vecy = self._get_field_seq(vecy)

        self._skip = skip = self._compute_skip_quiver()

        if XX is None and YY is None:
            [XX, YY] = np.meshgrid(self.oper.x_seq, self.oper.y_seq)

        if mpi.rank == 0:
            normalize_diff = (
                np.max(np.sqrt(vecx**2 + vecy**2)) -
                np.min(np.sqrt(vecx**2 + vecy**2))) / (np.max(
                    np.sqrt(vecx**2 + vecy**2)))
            vmax = np.max(np.sqrt(vecx**2 + vecy**2))
            # Quiver is normalized by the vmax
            # copy to avoid a bug
            vecx_c = vecx[::skip, ::skip].copy()
            vecy_c = vecy[::skip, ::skip].copy()
            quiver = ax.quiver(
                XX[::skip, ::skip],
                YY[::skip, ::skip],
                vecx_c/vmax, vecy_c/vmax)
        else:
            quiver = vmax = None

        return quiver, vmax
