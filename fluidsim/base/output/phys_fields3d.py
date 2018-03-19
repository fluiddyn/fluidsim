"""Physical fields output 3d (:mod:`fluidsim.base.output.phys_fields3d`)
========================================================================

Provides:

.. autoclass:: MoviesBasePhysFields3D
   :members:
   :private-members:

.. autoclass:: PhysFieldsBase3D
   :members:
   :private-members:

"""

from past.builtins import basestring

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from .phys_fields2d import (
    MoviesBasePhysFields2D, PhysFieldsBase2D)


class MoviesBasePhysFields3D(MoviesBasePhysFields2D):
    pass


class PhysFieldsBase3D(PhysFieldsBase2D):
    def _init_movies(self):
        self.movies = MoviesBasePhysFields3D(self.output, self)

    def set_equation_crosssection(self, equation):
        """Set the equation defining the cross-section.

        Parameters
        ----------

        equation : str

          The equation can be of the shape 'iz=2', 'z=1', ...

        """
        self._equation = equation

    def plot(self, field=None, time=None,
             QUIVER=True, vector='v', equation='iz=0',
             nb_contours=20, type_plot='contourf', vmin=None, vmax=None,
             cmap='viridis', numfig=None):
        """Plot a field.

        Parameters
        ----------

        field : {str, np.ndarray}, optional

        time : number, optional

        QUIVER : True

        vecx : 'ux'

        vecy : 'uy'

        nb_contours : 20

        type_plot : 'contourf'

        vmin : None

        vmax : None

        cmap : 'viridis'

        numfig : None

        """

        is_field_ready = False

        self._has_uxuy = self.sim.state.has_vars('vx', 'vy')

        key_field = None
        if field is None:
            key_field = self.field_to_plot
        elif isinstance(field, np.ndarray):
            key_field = 'given array'
            is_field_ready = True
        elif isinstance(field, basestring):
            key_field = field

        assert key_field is not None

        if equation.startswith('iz=') or equation.startswith('z='):
            vecx = vector + 'x'
            vecy = vector + 'y'
        elif equation.startswith('iy=') or equation.startswith('y='):
            vecx = vector + 'x'
            vecy = vector + 'z'
        elif equation.startswith('ix=') or equation.startswith('x='):
            vecx = vector + 'y'
            vecy = vector + 'z'
        else:
            raise NotImplementedError

        keys_state_phys = self.sim.state.keys_state_phys
        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

        if time is None and not is_field_ready:
            # we have to get the field from the state
            time = self.sim.time_stepping.t
            field, _ = self.get_field_to_plot_from_state(
                key_field, equation=equation)
            if QUIVER:
                vecx, _ = self.get_field_to_plot_from_state(
                    vecx, equation=equation)
                vecy, _ = self.get_field_to_plot_from_state(
                    vecy, equation=equation)
        else:
            # we have to get the field from a file
            self.set_of_phys_files.update_times()
            if key_field not in self.sim.state.keys_state_phys:
                raise ValueError('key not in state.keys_state_phys')

            field = self.get_field_to_plot(
                key=key_field, time=time, equation=equation)
            if QUIVER:
                vecx = self.get_field_to_plot(
                    key=vecx, time=time, equation=equation)
                vecy = self.get_field_to_plot(
                    key=vecy, time=time, equation=equation)

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe()
            else:
                fig, ax = self.output.figure_axe(numfig=numfig)

            if equation.startswith('iz=') or equation.startswith('z='):
                x_seq = self.oper.x_seq
                y_seq = self.oper.y_seq
            elif equation.startswith('iy=') or equation.startswith('y='):
                x_seq = self.oper.x_seq
                y_seq = self.oper.z_seq
            elif equation.startswith('ix=') or equation.startswith('x='):
                x_seq = self.oper.y_seq
                y_seq = self.oper.z_seq
            else:
                raise NotImplementedError

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
