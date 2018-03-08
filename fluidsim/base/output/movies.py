"""Movies output (:mod:`fluidsim.base.output.movies`)
=====================================================

Contains base classes which acts as a framework to
implement the method `animate` to make movies.


Provides:

.. autoclass:: MoviesBase
   :members:
   :private-members:

.. autoclass:: MoviesBase1D
   :members:
   :private-members:

.. autoclass:: MoviesBase2D
   :members:
   :private-members:

"""

from __future__ import division
from __future__ import print_function
from builtins import object
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from fluiddyn.util import mpi, is_run_from_jupyter
from fluiddyn.io import FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH, stdout_redirected
from fluidsim.util.util import pathdir_from_namedir


class MoviesBase(object):
    """Base class defining most generic functions for movies."""

    def __init__(self, output, **kwargs):
        params = output.sim.params
        self.output = output
        self.sim = output.sim
        self.params = params.output
        self.oper = self.sim.oper

        self._ani_fig = None
        self._ani_t = None
        self._ani_t_actual = None

    def _ani_init(self, key_field, numfig, dt_equations, tmin, tmax,
                  fig_kw, **kwargs):
        """Replace this function to initialize animation data and figure to
        plot on.

        """
        pass

    def _ani_update(self, frame, **fargs):
        """Replace this function to load data for next frame and update the
        figure.

        """
        pass

    def _set_path(self):
        """Sets path attribute specifying the location of saved files.
        Tries different directories in the following order:
        FLUIDSIM_PATH, FLUIDDYN_PATH_SCRATCH, output.path_run

        .. TODO:Need to check if pathdir_from_namedir call is read.
        """
        self.path = os.path.join(FLUIDSIM_PATH, self.sim.name_run)

        if not os.path.exists(self.path):

            if FLUIDDYN_PATH_SCRATCH is not None:
                self.path = os.path.join(
                    FLUIDDYN_PATH_SCRATCH, self.sim.name_run)
                if not os.path.exists(self.path):
                    self.path = self.output.path_run
            else:
                self.path = self.output.path_run
            if not os.path.exists(self.path):
                self.path = None

        self.path = pathdir_from_namedir(self.path)

    def _set_font(self, family='serif', size=12):
        """Use to set font attribute. May be either an alias (generic name
        is CSS parlance), such as serif, sans-serif, cursive, fantasy, or
        monospace, a real font name or a list of real font names.

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
        """Replace this function to change the default axes set while
        animating.

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

    def _select_key_field(self, key_field):
        """
        Defines key_field default.
        """
        # Compute keys of the simulation.
        keys_state_phys = self.sim.info.solver.classes.State['keys_state_phys']
        keys_computable = self.sim.info.solver.classes.State['keys_computable']

        if key_field is None:
            field_to_plot = self.params.output.phys_fields.field_to_plot
            if field_to_plot in keys_state_phys or \
               field_to_plot in keys_computable:
                key_field = field_to_plot
            else:
                raise ValueError(
                    'params.output.phys_fields.field_to_plot not '
                    'in keys_state_phys')
        else:
            if (key_field in keys_state_phys or key_field in keys_computable):
                key_field = key_field

            else:
                raise ValueError('key_field not in keys_state_phys')

        return key_field

    def animate(
            self, key_field=None, dt_frame_in_sec=0.3, dt_equations=None,
            tmin=None, tmax=None, repeat=True, save_file=False, numfig=None,
            fargs=None, fig_kw=None, **kwargs):
        """
        Load the key field from multiple save files and display as
        an animated plot or save as a movie file.

        Parameters
        ----------
        key_field : str
            Specifies which field to animate
        dt_frame_in_sec : float
            Interval between animated frames in seconds
        dt_equations : float
            Approx. interval between saved files to load in simulation time
            units
        tmax : float
            Animate till time `tmax`.
        repeat : bool
            Loop the animation
        save_file : str or bool
            Path to save the movie. When `True`  saves into a file instead
            of plotting it on screen (default: ~/fluidsim_movie.mp4). Specify
            a string to save to another file location. Format is autodetected
            from the filename extension.
        numfig : int
            Figure number on the window
        fargs : dict
            Dictionary of arguments for `_ani_update`. Matplotlib requirement.
        fig_kw : dict
            Dictionary of arguments for arguments for the figure.

        Keyword Parameters
        ------------------
        All `kwargs` are passed on to `_ani_init` and `_ani_save`

        xmax : float
            Set x-axis limit for 1D animated plots
        ymax : float
            Set y-axis limit for 1D animated plots
        clim : tuple
            Set colorbar limits for 2D animated plots
        step : int
            Set step value to get a coarse 2D field
        QUIVER : bool
            Set quiver on or off on top of 2D pcolor plots
        codec : str
            Codec used to save into a movie file (default: ffmpeg)

        Examples
        --------
        >>> import fluidsim as fls
        >>> sim = fls.load_sim_for_plot()
        >>> animate = sim.output.spectra.animate
        >>> animate('E')
        >>> animate('rot')
        >>> animate('rot', dt_equations=0.1, dt_frame_in_sec=50, clim=(-5,5))
        >>> animate('rot', tmax=25, clim=(-5,5), save_file='True')
        >>> animate('rot', clim=(-5,5), save_file='~/fluidsim.gif', codec='imagemagick')

        .. TODO: Use FuncAnimation with blit=True option.

        Note
        ----
        This method is kept as generic as possible. Any arguments specific to
        1D, 2D or 3D animations or specific to a type of output are be passed
        via keyword arguments (``kwargs``) into its respective ``_ani_init`` or
        ``_ani_update`` methods.

        """
        if mpi.rank > 0:
            raise NotImplementedError('Do NOT use this script with MPI !')

        if fargs is None:
            fargs = {}

        if fig_kw is None:
            fig_kw = {}

        self._ani_init(key_field, numfig, dt_equations, tmin, tmax,
                       fig_kw, **kwargs)
        if is_run_from_jupyter():
            try:
                from ipywidgets import interact, widgets
                slider = widgets.IntSlider(
                    min=tmin, max=tmax, step=dt_equations, value=tmin)

                def widget_update(time):
                    self._ani_update(time)
                    self._ani_fig.canvas.draw()

                interact(widget_update, time=slider)
            except ImportError:
                print('Install jupyter nbextension ipywidgets and enable it:\n'
                      '  pip install ipywidgets\n'
                      '  jupyter-nbextension enable --py widgetsnbextension\n'
                      'Restart the notebook and call the function under the'
                      'magic:\n'
                      '  %matplotlib notebook')
        else:
            self._animation = animation.FuncAnimation(
                self._ani_fig, self._ani_update, len(self._ani_t),
                fargs=fargs.items(), interval=dt_frame_in_sec * 1000,
                blit=False, repeat=repeat)

        if save_file:
            if not isinstance(save_file, str):
                save_file = r'~/fluidsim_movie.mp4'

            self._ani_save(save_file, dt_frame_in_sec, **kwargs)

    def _ani_save(self, path_file, dt_frame_in_sec, codec='ffmpeg', **kwargs):
        """Saves the animation using `matplotlib.animation.writers`."""

        path_file = os.path.expandvars(path_file)
        path_file = os.path.expanduser(path_file)
        avail = animation.writers.avail
        if len(avail) == 0:
            raise ValueError(
                'Please install a codec library. For e.g. ffmpeg, mencoder, '
                'imagemagick, html')
        elif codec not in avail:
            print(
                'Using one of the available codecs: {}'.format(avail.keys()))
            codec = list(avail.keys())[0]

        Writer = animation.writers[codec]

        print('Saving movie to ', path_file, '...')
        writer = Writer(fps=1. / dt_frame_in_sec,
                        metadata=dict(artist='FluidSim'))
        # _animation is a FuncAnimation object
        self._animation.save(path_file, writer=writer, dpi=150)

    def _ani_get_t_actual(self, time):
        '''Find the index and value of the closest actual time of the field.'''
        idx = np.abs(self._ani_t_actual - time).argmin()
        return idx, self._ani_t_actual[idx]


class MoviesBase1D(MoviesBase):
    """Base class defining most generic functions for movies for 1D data."""

    def _ani_init(self, key_field, numfig, dt_equations, tmin, tmax,
                  fig_kw, **kwargs):
        """Initializes animated fig. and list of times of save files to load."""

        if key_field is None:
            raise ValueError('key_field should not be None.')

        self._ani_key = key_field
        self._ani_fig, self._ani_ax = plt.subplots(num=numfig, **fig_kw)
        self._ani_line, = self._ani_ax.plot([], [])
        self._ani_t = []
        self._set_font()
        self._set_path()

        ax = self._ani_ax
        ax.set_xlim(0, kwargs['xmax'])
        ax.set_ylim(10e-16, kwargs['ymax'])

        if tmax is None:
            tmax = int(self.sim.time_stepping.t)

        if dt_equations is None:
            dt_equations = self.params.periods_save.phys_fields

        self._ani_t = list(np.arange(0, tmax + dt_equations, dt_equations))

    def _ani_update(self, frame, **fargs):
        """Loads contour data and updates figure."""

        time = self._ani_t[frame]
        with stdout_redirected():
            y, key_field = self._ani_get_field(time)
        x = self._select_axis()

        self._ani_line.set_data(x, y)
        title = (key_field +
                 ', $t = {0:.3f}$, '.format(time) +
                 self.output.name_solver +
                 ', $n_x = {0:d}$'.format(self.params.oper.nx))
        self._ani_ax.set_title(title)

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
    """Base class defining most generic functions for movies for 2D data."""

    def _ani_init(self, key_field, numfig, dt_equations, tmin, tmax,
                  fig_kw, **kwargs):
        """Initializes animated fig. and list of times of save files to load.
        """

        self._ani_key = self._select_key_field(key_field)
        self._ani_t = []
        self._set_font()
        self._set_path()

        if tmax is None:
            tmax = self.sim.time_stepping.t

        if tmin is None:
            tmin = 0

        if tmin > tmax:
            raise ValueError('Error tmin > tmax. '
                             'Value tmin should be smaller than tmax')

        if dt_equations is None:
            dt_equations = self.params.periods_save.phys_fields

        self._ani_t = list(np.arange(tmin, tmax + dt_equations, dt_equations))

        self._ani_fig, self._ani_ax = plt.subplots(num=numfig, **fig_kw)

    def _select_axis(self, xlabel='x', ylabel='y'):
        '''Get 1D arrays for setting the axes.'''
        x = self.oper.x_seq
        y = self.oper.y_seq
        self._ani_ax.set_xlabel(xlabel, fontdict=self.font)
        self._ani_ax.set_ylabel(ylabel, fontdict=self.font)
        return x, y
