from __future__ import division
from __future__ import print_function
from builtins import object
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from fluiddyn.util import mpi
from fluiddyn.io import FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH, stdout_redirected
from fluidsim.util.util import pathdir_from_namedir


class MoviesBase(object):

    def _set_path(self):
        """
        Sets path attribute specifying the location of saved files.
        Tries different directories int the following order:
        FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH, output.path_run

        .. TODO:Need to check if pathdir_from_namedir call is reqd.
        """
        self.path = os.path.join(FLUIDSIM_PATH, self.sim.name_run)

        if not os.path.exists(self.path):
            self.path = os.path.join(
                FLUIDDYN_PATH_SCRATCH, self.sim.name_run)
            if not os.path.exists(self.path):
                self.path = self.output.path_run
                if not os.path.exists(self.path):
                    self.path = None

        self.path = pathdir_from_namedir(self.path)

    def _set_font(self, family='serif', size=12):
        """
        Use to set font attribute. May be either an alias (generic name is CSS parlance), such as
        serif, sans-serif, cursive, fantasy, or monospace, a real font name or a list of real font names.
        """

        self.font = {'family': family,
                     'color': 'black',
                     'weight': 'normal',
                     'size': size}

    def _ani_get_field(self, time):
        """
        Loads a saved file corresponding to an approx. time, and
        returns contour data
        """
        raise NotImplementedError

    def _select_axis(self, xlabel='x', ylabel='y'):
        """
        Replace this function to change the default axes set while animating.
        """
        pass

    def _select_field(self, field=None, key_field=None):
        """
        Once a saved file is loaded, this selects the field and mpi-gathers.

        Returns
        -------
        field : nd array
        key_field : string
        """
        raise NotImplementedError('_select_field function declaration missing')

    def animate(self, key_field=None, numfig=None, frame_dt=300, file_dt=1,
                tmax=None, repeat=True, save_file=None, fargs={}, **kwargs):
        """
        Load the key field from multiple save files and display as
        an animated plot or save as a movie file.

        Parameters
        ----------
        key_field : str
            Specifies which field to animate
        numfig : int
            Figure number on the window
        frame_dt : int
            Interval between animated frames in milliseconds
        file_dt : float
            Approx. interval between saved files to load in simulation time
            units
        tmax : float
            Animate till time `tmax`.
        repeat : bool
            Loop the animation
        save_file : str
            When not `None` saves into a file instead of plotting it on screen
        fargs : dict
            Dictionary of arguments for `_ani_update`. Matplotlib requirement.

        Keyword Parameters
        ------------------
        All `kwargs` are passed on to `_ani_init`.

        xmax : float
            Set x-axis limit for 1D animated plots
        ymax : float
            Set y-axis limit for 1D animated plots
        clim : tuple
            Set colorbar limit for 2D animated plots

        Examples
        --------
        >>> sim.output.spectra.animate('E')
        >>> sim.output.phys_fields.animate('rot')
        >>> sim.output.phys_fields.animate('rot', file_dt=0.1, frame_dt=50, clim=(-5,5))

        .. TODO: Use FuncAnimation with blit=True option.

        """

        if mpi.nb_proc > 1:
            raise ValueError('Do NOT use this script with MPI !\n'
                             'The MPI version of get_state_from_simul()\n'
                             'is not implemented.')

        self._ani_init(key_field, numfig, file_dt, tmax, **kwargs)
        self._animation = animation.FuncAnimation(
            self._ani_fig, self._ani_update, len(self._ani_t), fargs=fargs.items(),
            interval=frame_dt, blit=False, repeat=repeat)

        if save_file is not None:
            self._ani_save(save_file, frame_dt)

    def _ani_init(self, **kwargs):
        pass

    def _ani_update(self, **fargs):
        pass

    def _ani_save(self, filename=r'movie.avi', frame_dt=300):
        try:
            Writer = animation.writers['ffmpeg']
        except KeyError:
            Writer = animation.writers['mencoder']

        print(('Saving movie to ', filename, '...'))
        writer = Writer(
            fps=1000. / frame_dt, metadata=dict(artist='Me'), bitrate=1800)
        self._animation.save(filename, writer=writer)  # _animation is a FuncAnimation object

    def _ani_get_t_actual(self, time):
        '''Find the index and value of the closest actual time of the field.'''
        idx = np.abs(self._ani_t_actual - time).argmin()
        return idx, self._ani_t_actual[idx]


class MoviesBase1D(MoviesBase):

    def _ani_init(self, key_field, numfig, file_dt, tmax, **kwargs):
        """Initializes animated fig. and list of times of save files to load."""

        self._ani_key = key_field
        self._ani_fig, self._ani_ax = plt.subplots(num=numfig)
        self._ani_line, = self._ani_ax.plot([], [])
        self._ani_t = []
        self._set_font()
        self._set_path()

        ax = self._ani_ax
        ax.set_xlim(0, kwargs['xmax'])
        ax.set_ylim(10e-16, kwargs['ymax'])

        if tmax is None:
            tmax = int(self.sim.time_stepping.t)

        self._ani_t = list(np.arange(0, tmax + file_dt, file_dt))

    def _ani_update(self, frame, **fargs):
        """Loads contour data and updates figure."""

        time = self._ani_t[frame]
        with stdout_redirected():
            y, key_field = self._ani_get_field(time)
        x = self._select_axis()

        self._ani_line.set_data(x, y)
        title = (key_field +
                 ', t = {0:.3f}, '.format(time) +
                 self.output.name_solver +
                 ', nh = {0:d}'.format(self.params.oper.nx))
        self._ani_ax.set_title(r'${}$'.format(title))

        return self._ani_line

    def _select_axis(self, xlabel='x', ylabel='u'):
        '''Get 1D arrays for setting the X-axis.'''
        try:
            x = self.oper.xs
        except AttributeError:
            x = self.oper.x_seq

        self._ani_ax.set_xlabel(xlabel, fontdict=self.font)
        self._ani_ax.set_ylabel(ylabel, fontdict=self.font)
        return x


class MoviesBase2D(MoviesBase):

    def _ani_init(self, key_field, numfig, file_dt, tmax, **kwargs):
        """ Initializes animated fig. and list of times of save files to load."""

        self._ani_key = key_field
        self._ani_t = []
        self._set_font()
        self._set_path()

        if tmax is None:
            tmax = int(self.sim.time_stepping.t)

        self._ani_t = list(np.arange(0, tmax + file_dt, file_dt))
        self._ani_fig, self._ani_ax = plt.subplots(num=numfig)

    def _select_axis(self, xlabel='x', ylabel='y'):
        '''Get 1D arrays for setting the axes.'''
        x = self.oper.x_seq
        y = self.oper.y_seq
        self._ani_ax.set_xlabel(xlabel, fontdict=self.font)
        self._ani_ax.set_ylabel(ylabel, fontdict=self.font)
        return x, y
