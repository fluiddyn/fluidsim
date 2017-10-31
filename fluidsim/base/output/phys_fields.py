"""Physical fields output (:mod:`fluidsim.base.output.phys_fields`)
=========================================================================

Provides:

.. autoclass:: PhysFieldsBase
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
import numpy as np
import h5py

from fluiddyn.util import mpi
from .base import SpecificOutput
from .movies import MoviesBase1D, MoviesBase2D
from ..params import Parameters

cfg_h5py = h5py.h5.get_config()


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

    def init_files(self, dico_arrays_1time=None):
        pass

    def init_online_plot(self):
        pass

    def online_save(self):
        """Online save."""
        tsim = self.sim.time_stepping.t
        if self._has_to_online_save():
            self.t_last_save = tsim
            self.save()

    def online_plot(self):
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
            name_save = 'state_phys_t={:07.3f}_it={}.hd5'.format(
                time, self.sim.time_stepping.it)
        else:
            name_save = 'state_phys_t={:07.3f}.hd5'.format(time)

        path_file = os.path.join(path_run, name_save)
        if os.path.exists(path_file):
            name_save = 'state_phys_t={:07.3f}_it={}.hd5'.format(
                time, self.sim.time_stepping.it)
            path_file = os.path.join(path_run, name_save)
        to_print = 'save state_phys in file ' + name_save
        self.output.print_stdout(to_print)

        if mpi.nb_proc == 1 or not cfg_h5py.mpi:
            if mpi.rank == 0:
                f = h5py.File(path_file, 'w')
                group_state_phys = f.create_group("state_phys")
                group_state_phys.attrs['what'] = 'obj state_phys for solveq2d'
                group_state_phys.attrs['name_type_variables'] = state_phys.info
                group_state_phys.attrs['time'] = time
                group_state_phys.attrs['it'] = self.sim.time_stepping.it
        else:
            f = h5py.File(path_file, 'w', driver='mpio', comm=mpi.comm)
            group_state_phys = f.create_group("state_phys")
            group_state_phys.attrs['what'] = 'obj state_phys for solveq2d'
            group_state_phys.attrs['name_type_variables'] = state_phys.info

            group_state_phys.attrs['time'] = time
            group_state_phys.attrs['it'] = self.sim.time_stepping.it

        if mpi.nb_proc == 1:
            for k in state_phys.keys:
                field_seq = state_phys.get_var(k)
                group_state_phys.create_dataset(k, data=field_seq)
        elif not cfg_h5py.mpi:
            for k in state_phys.keys:
                field_loc = state_phys.get_var(k)
                field_seq = self.oper.gather_Xspace(field_loc)
                if mpi.rank == 0:
                    group_state_phys.create_dataset(k, data=field_seq)
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
                f = h5py.File(path_file, 'w')

        if mpi.rank == 0:
            f.attrs['date saving'] = str(datetime.datetime.now()).encode()
            f.attrs['name_solver'] = self.output.name_solver
            f.attrs['name_run'] = self.output.name_run
            if particular_attr is not None:
                f.attrs['particular_attr'] = particular_attr

            self.sim.info._save_as_hdf5(hdf5_parent=f)
            gp_info = f['info_simul']
            gf_params = gp_info['params']
            gf_params.attrs['SAVE'] = True
            gf_params.attrs['NEW_DIR_RESULTS'] = True
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
                field_loc = self.sim.state(key_field)
            else:
                field_loc = self.sim.state(key_field)
        else:
            key_field = 'given field'
            field_loc = field

        if mpi.nb_proc > 1:
            field = self.oper.gather_Xspace(field_loc)
        else:
            field = field_loc

        return field, key_field


class PhysFieldsBase1D(PhysFieldsBase, MoviesBase1D):

    def plot(self, numfig=None, field=None, key_field=None):
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
    t = float(re.search('[-+]?[0-9]*\.?[0-9]+', filename).group(0))
    return t


class MoviesBasePhysFields2D(MoviesBase2D):
    """Methods required to animate physical fields HDF5 files."""

    def _ani_init(self, key_field, numfig, file_dt, tmin, tmax, **kwargs):
        """Initialize list of files and times, pcolor plot, quiver and colorbar.
        """
        self._set_path()
        self._ani_pathfiles = sorted(glob(os.path.join(
            self.path, 'state_phys*')))
        self._ani_t_actual = np.array(list(
            map(time_from_path, self._ani_pathfiles)))

        if tmax is None:
            tmax = self._ani_t_actual.max()

        super(MoviesBasePhysFields2D, self)._ani_init(
            key_field, numfig, file_dt, tmin, tmax, **kwargs)

        field, ux, uy = self._ani_get_field(0)
        x, y = self._select_axis(shape=ux.shape)
        XX, YY = np.meshgrid(x, y)

        self._ani_im = self._ani_ax.pcolor(XX, YY, field)
        self._ani_cbar = self._ani_fig.colorbar(self._ani_im)
        self._ani_clim = kwargs.get('clim')
        self._ani_set_clim()
        self._ani_quiver, vmax = self._quiver_plot(
            self._ani_ax, ux, uy, XX, YY)

    def _quiver_plot(self, ax, vecx='ux', vecy='uy', XX=None, YY=None):
        '''Make a quiver plot on axis `ax`.'''
        pass

    def _select_axis(self, xlabel='x', ylabel='y', shape=None):
        '''Get 1D arrays for setting the axes.'''

        x, y = super(MoviesBasePhysFields2D, self)._select_axis(xlabel, ylabel)
        if shape is not None and (x.shape[0], y.shape[0]) != shape:
            path_file = os.path.join(self.path, 'params_simul.xml')
            params = Parameters(path_file=path_file)
            x = np.arange(0, params.oper.Lx, params.oper.nx)
            y = np.arange(0, params.oper.Ly, params.oper.ny)

        return x, y

    def _ani_get_field(self, time):
        """Get field, ux, uy from saved physical fields."""

        idx, t_actual = self._ani_get_t_actual(time)

        with h5py.File(self._ani_pathfiles[idx]) as f:
            field = f['state_phys'][self._ani_key].value
            ux = f['state_phys']['ux'].value
            uy = f['state_phys']['uy'].value

        return field, ux, uy

    def _ani_update(self, frame, **fargs):
        """Loads data and updates figure."""

        time = self._ani_t[frame]

        field, ux, uy = self._ani_get_field(time)
        field = field[:-1, :-1]

        # Update figure, quiver and colorbar
        self._ani_im.set_array(field.flatten())

        self._ani_quiver.set_UVC(ux[::self._skip, ::self._skip],
                                 uy[::self._skip, ::self._skip])
        self._ani_im.autoscale()
        self._ani_set_clim()

        idx, time = self._ani_get_t_actual(time)
        title = (self._ani_key +
                 ', $t = {0:.3f}$, '.format(time) +
                 self.output.name_solver +
                 ', $n_x = {0:d}$'.format(self.params.oper.nx))

        vmax = np.max(np.sqrt(ux ** 2 + uy ** 2))
        title += r', $|\vec{v}|_{max} = $' + '{0:.3f}'.format(vmax)

        self._ani_ax.set_title(title)

    def _ani_set_clim(self):
        """Maintains a constant colorbar throughout the animation."""

        clim = self._ani_clim
        if clim is not None:
            self._ani_im.set_clim(*clim)
            self._ani_cbar.set_clim(*clim)
            ticks = np.linspace(*clim, num=21, endpoint=True)
            self._ani_cbar.set_ticks(ticks)


class PhysFieldsBase2D(PhysFieldsBase, MoviesBasePhysFields2D):

    def plot(self, numfig=None, field=None, key_field=None,
             QUIVER=True, vecx='ux', vecy='uy',
             nb_contours=20, type_plot='contourf', iz=0,
             vmin=None, vmax=None, cmap='viridis'):

        field, key_field = self._select_field(field, key_field)
        keys_state_phys = self.sim.state.keys_state_phys
        x_left_axe = 0.08
        z_bottom_axe = 0.1
        width_axe = 0.95
        height_axe = 0.83
        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]

        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

        if field.ndim == 3:
            field = field[iz]

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe(size_axe=size_axe)
            else:
                fig, ax = self.output.figure_axe(numfig=numfig,
                                                 size_axe=size_axe)
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

        if mpi.rank == 0:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            title = (key_field +
                     ', $t = {0:.3f}$, '.format(self.sim.time_stepping.t) +
                     self.output.name_solver +
                     ', $n_x = {0:d}$'.format(self.params.oper.nx))

            if QUIVER:
                title += r', $max(|v|) = {0:.3f}$'.format(vmax)

            ax.set_title(title)

            fig.canvas.draw()

    def _quiver_plot(self, ax, vecx='ux', vecy='uy', XX=None, YY=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        if isinstance(vecx, basestring):
            vecx_loc = self.sim.state(vecx)
            if mpi.nb_proc > 1:
                vecx = self.oper.gather_Xspace(vecx_loc)
            else:
                vecx = vecx_loc

        if isinstance(vecy, basestring):
            vecy_loc = self.sim.state(vecy)
            if mpi.nb_proc > 1:
                vecy = self.oper.gather_Xspace(vecy_loc)
            else:
                vecy = vecy_loc

        # 4% of the Lx it is a great separation between vector arrows.
        delta_quiver = 0.04 * self.oper.Lx
        skip = (self.oper.nx_seq / self.oper.Lx) * delta_quiver
        skip = int(np.round(skip))

        if skip < 1:
            skip = 1

        self._skip = skip

        if XX is None and YY is None:
            [XX, YY] = np.meshgrid(self.oper.x_seq, self.oper.y_seq)

        if mpi.rank == 0:
            # copy to avoid a bug
            vecx_c = vecx[::skip, ::skip].copy()
            vecy_c = vecy[::skip, ::skip].copy()
            quiver = ax.quiver(
                XX[::skip, ::skip],
                YY[::skip, ::skip],
                vecx_c, vecy_c)

        return quiver, np.max(np.sqrt(vecx**2 + vecy**2))
