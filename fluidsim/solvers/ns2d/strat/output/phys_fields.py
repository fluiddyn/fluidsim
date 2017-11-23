"""Physical fields output (:mod:`fluidsim.solvers.ns2d.strat.output.phys_fields`)
================================================================================

Provides:

.. autoclass:: PhysFields2DStrat
   :members:
   :private-members:

"""

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi
from fluidsim.base.output.phys_fields import PhysFieldsBase2D


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
