"""Physical fields output (:mod:`fluidsim.base.output.phys_fields`)
=========================================================================

.. currentmodule:: fluidsim.base.output.phys_fields

Provides:

.. autoclass:: PhysFieldsBase
   :members:
   :private-members:

"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import datetime

from fluiddyn.util import mpi
from .base import SpecificOutput
from .movies import MoviesBase1D, MoviesBase2D


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
        if (tsim-self.t_last_plot >= self.period_plot):
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

        if mpi.rank == 0:
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

            f = h5py.File(path_file, 'w')
            f.attrs['date saving'] = str(datetime.datetime.now())
            f.attrs['name_solver'] = self.output.name_solver
            f.attrs['name_run'] = self.output.name_run
            if particular_attr is not None:
                f.attrs['particular_attr'] = particular_attr

            self.sim.info._save_as_hdf5(hdf5_parent=f)
            gp_info = f['info_simul']
            gf_params = gp_info['params']
            gf_params.attrs['SAVE'] = True
            gf_params.attrs['NEW_DIR_RESULTS'] = True

            group_state_phys = f.create_group("state_phys")
            group_state_phys.attrs['what'] = 'obj state_phys for solveq2d'
            group_state_phys.attrs['name_type_variables'] = (
                state_phys.info)

            group_state_phys.attrs['time'] = time
            group_state_phys.attrs['it'] = self.sim.time_stepping.it

        for k in state_phys.keys:
            field_loc = state_phys.get_var(k)
            if mpi.nb_proc > 1:
                field_seq = self.oper.gather_Xspace(field_loc)
            else:
                field_seq = field_loc
            if mpi.rank == 0:
                group_state_phys.create_dataset(k, data=field_seq)

        if mpi.rank == 0:
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
        field, key_field = self._field_to_plot(field, key_field)

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe(size_axe=None)
            else:
                fig, ax = self.output.figure_axe(numfig=numfig,
                                                 size_axe=None)
            xs = self.oper.xs

            ax.plot(xs, field)


class PhysFieldsBase2D(PhysFieldsBase, MoviesBase2D):

    def plot(self, numfig=None, field=None, key_field=None,
             QUIVER=True, vecx='ux', vecy='uy', FIELD_LOC=True,
             nb_contours=20, type_plot='contourf', iz=0,
             cmap='viridis'):

        field, key_field = self._select_field(field, key_field)
        keys_state_phys = self.sim.state.keys_state_phys
        x_left_axe = 0.08
        z_bottom_axe = 0.07
        width_axe = 0.97
        height_axe = 0.87
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
                contours = ax.contourf(x_seq, y_seq, field,
                                       nb_contours, cmap=cmap)
                fig.colorbar(contours)
                fig.contours = contours
            elif type_plot == 'pcolor':
                pc = ax.pcolormesh(x_seq, y_seq, field,
                                   cmap=cmap)
                fig.colorbar(pc)

        if QUIVER:
            vmax = self._quiver_plot(ax, vecx, vecy)

        if mpi.rank == 0:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            title = (key_field +
                     ', t = {0:.3f}, '.format(self.sim.time_stepping.t) +
                     self.output.name_solver +
                     ', nh = {0:d}'.format(self.params.oper.nx))

            if QUIVER:
                title += r', max(|v|) = {0:.3f}'.format(vmax)

            ax.set_title(title)

            fig.canvas.draw()

    def _quiver_plot(self, ax, vecx='ux', vecy='uy'):
        """
        Superimposes a quiver plot of velocity vectors
        with a given ax object corresponding to
        a 2D contour plot.
        """

        if isinstance(vecx, str):
            vecx_loc = self.sim.state(vecx)
            if mpi.nb_proc > 1:
                vecx = self.oper.gather_Xspace(vecx_loc)
            else:
                vecx = vecx_loc
        if isinstance(vecy, str):
            vecy_loc = self.sim.state(vecy)
            if mpi.nb_proc > 1:
                vecy = self.oper.gather_Xspace(vecy_loc)
            else:
                vecy = vecy_loc
        pas_vector = np.round(self.oper.nx_seq / 48)
        if pas_vector < 1:
            pas_vector = 1

        [XX_seq, YY_seq] = np.meshgrid(self.oper.x_seq, self.oper.y_seq)
        if mpi.rank == 0:
            # copy to avoid a bug
            vecx_c = vecx[::pas_vector, ::pas_vector].copy()
            vecy_c = vecy[::pas_vector, ::pas_vector].copy()
            ax.quiver(XX_seq[::pas_vector, ::pas_vector],
                      YY_seq[::pas_vector, ::pas_vector],
                      vecx_c, vecy_c)

        return np.max(np.sqrt(vecx**2 + vecy**2))
