"""Physical fields output (:mod:`fluidsim.solvers.ns2d.strat.output.phys_fields`)
================================================================================

Provides:

.. autoclass:: PhysFields2DStrat
   :members:
   :private-members:

"""
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from past.builtins import basestring

from fluiddyn.util import mpi
from fluidsim.base.output.phys_fields import PhysFieldsBase2D
from fluidsim.base.output.phys_fields import time_from_path


class PhysFields2DStrat(PhysFieldsBase2D):

    def _ani_update(self, frame, **fargs):

        super(PhysFields2DStrat, self)._ani_update(frame, **fargs)

        time = self._ani_t[frame]
        title = (self._ani_key +
                 ', R = {0:.2f}'.format(
                     self.output.ratio_omegas) +
                 ', F = {0:.1f}'.format(
                     self.output.froude_number) +
                 ', $t = {0:.3f}$'.format(time))

        self._ani_ax.set_title(title)

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
                     ', R = {0:.2f}'.format(
                         self.output.ratio_omegas) +
                     ', F = {0:.1f}'.format(
                         self.output.froude_number) +
                     ', $t = {0:.3f}$ '.format(
                         self.sim.time_stepping.t))

            if QUIVER:
                title += r', $max(|v|) = {0:.3f}$'.format(vmax)

            ax.set_title(title)

            fig.canvas.draw()


    def _plot_init(self, key_field):
        """
        Initializes the plot of the physical fields.
        """
        self._set_path()
        if key_field is None:
            raise ValueError('key_field should not be None.')

        self._ani_key = key_field

        if self._ani_key not in self.sim.state.keys_state_phys:
            raise ValueError('key not in state.keys_state_phys')

        self._ani_pathfiles = sorted(glob(os.path.join(
            self.path, 'state_phys*')))
        self._ani_t_actual = np.array(list(
            map(time_from_path, self._ani_pathfiles)))


    def plot_inlet(self, time=None, key_field='rot', INLET_PLOT=True,
                   numfig=None, QUIVER=True, vecx='ux', vecy='uy',
                   nb_contours=20, type_plot='contourf', iz=0,
                   vmin=None, vmax=None, cmap='viridis'):
        """
        Plots the physical fields with an inlet windows.
        """
        if time is None:
            time = self.sim.time_stepping.t

        # Init the plot
        self._plot_init(key_field=key_field)

        # Get the field
        idx, t_actual = self._ani_get_t_actual(time)
        field, ux, uy = self._ani_get_field(time)

        # Parameters of the figure
        x_left_axe = 0.08
        z_bottom_axe = 0.1
        width_axe = 0.95
        height_axe = 0.83
        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]

        keys_state_phys = self.sim.state.keys_state_phys
        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

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

                # INLET PLOT
                if INLET_PLOT:
                    self._inlet_plot(fig, time)

            elif type_plot == 'pcolor':
                pc = ax.pcolormesh(x_seq, y_seq, field,
                                   vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(pc)
        else:
            ax = None

        if QUIVER:
            quiver, vmax = self._quiver_plot(ax, ux, uy)

        if mpi.rank == 0:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            title = (self._ani_key +
                     ', R = {0:.2f}'.format(
                         self.output.ratio_omegas) +
                     ', F = {}'.format(
                         self.output.froude_number) +
                     ', $t = {0:.3f}$ '.format(
                         t_actual))

            ax.set_title(title)

            fig.canvas.draw()

    def _quiver_plot(self, ax, ux, uy, vecx='ux', vecy='uy', XX=None, YY=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        if isinstance(vecx, basestring):
            vecx_loc = ux
            if mpi.nb_proc > 1:
                vecx = self.oper.gather_Xspace(vecx_loc)
            else:
                vecx = vecx_loc

        if isinstance(vecy, basestring):
            vecy_loc = uy
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


    def _get_field(self, key_spatial='E'):
        """ Get field for the inlet plot."""

        # Check if key_spatial can be loaded.
        keys_spatial = ['E', 'EK', 'EA']
        if key_spatial not in keys_spatial:
            raise ValueError('key_spatial not in spatial means keys.')

        # Load data for inlet plot
        dico = self.output.spatial_means.load()
        t = dico['t']
        E = dico[key_spatial]

        return t, E

    def _inlet_plot(self, fig, time):
        """ It makes the inlet plot. """
        t, E = self._get_field()
        idx_spatial = np.abs(t - time).argmin()
        t_inlet = t[0:idx_spatial]
        E_inlet = E[0:idx_spatial]
        left, bottom, width, height = [0.63, 0.72, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(t, E, linestyle='--', linewidth=1, color='grey', alpha=0.5)
        ax2.plot(t_inlet, E_inlet, linewidth=1, color='black')
        return ax2
