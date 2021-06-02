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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from fluiddyn.util import mpi, is_run_from_jupyter


class MoviesBase:
    """Base class defining most generic functions for movies."""

    def __init__(self, output):
        params = output.sim.params
        self.output = output
        self.sim = output.sim
        self.params = params.output
        self.oper = self.sim.oper

        self._set_font()

    def init_animation(
        self, key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
    ):
        """Initializes animated fig. and list of times of save files to load."""
        self._set_key_field(key_field)
        self._init_ani_times(tmin, tmax, dt_equations)
        self.fig, self.ax = plt.subplots(num=numfig, **fig_kw)
        self._init_labels()

    def _init_ani_times(self, tmin, tmax, dt_equations):
        """Initialization of the variable ani_times for one animation."""
        self.phys_fields.set_of_phys_files.update_times()
        if tmax is None:
            tmax = self.sim.time_stepping.t

        if tmin is None:
            tmin = self.phys_fields.set_of_phys_files.get_min_time()

        if tmin > tmax:
            raise ValueError(
                "Error tmin > tmax. Value tmin should be smaller than tmax"
            )

        if dt_equations is None:
            dt_equations = self.params.periods_save.phys_fields

        self.ani_times = np.arange(tmin, tmax + dt_equations, dt_equations)

    def update_animation(self, frame, **fargs):
        """Replace this function to load data for next frame and update the
        figure.

        """
        pass

    def _set_font(self, family="serif", size=12):
        """Use to set font attribute. May be either an alias (generic name
        is CSS parlance), such as serif, sans-serif, cursive, fantasy, or
        monospace, a real font name or a list of real font names.

        """
        self.font = {
            "family": family,
            "color": "black",
            "weight": "normal",
            "size": size,
        }

    def get_field_to_plot(self, time=None, key=None, equation=None):
        """
        Once a saved file is loaded, this selects the field and mpi-gathers.

        Returns
        -------
        field : nd array or string
        """
        raise NotImplementedError(
            "get_field_to_plot function declaration missing."
        )

    def _init_labels(self, xlabel=None, ylabel=None):
        """Initialize the labels."""
        if xlabel is None:
            xlabel = self.sim.oper.axes[1]
        if ylabel is None:
            ylabel = self.sim.oper.axes[0]
        self.ax.set_xlabel(xlabel, fontdict=self.font)
        self.ax.set_ylabel(ylabel, fontdict=self.font)

    def _get_axis_data(self):
        """Replace this function to load axis data."""
        raise NotImplementedError("_get_axis_data  function declaration missing.")

    def _set_key_field(self, key_field):
        """
        Defines key_field default.
        """
        self.key_field = self.phys_fields.get_key_field_to_plot(
            forbid_compute=True, key_field_to_plot=key_field
        )

    def animate(
        self,
        key_field=None,
        dt_frame_in_sec=0.3,
        dt_equations=None,
        tmin=None,
        tmax=None,
        repeat=True,
        save_file=False,
        numfig=None,
        fargs={},
        fig_kw={},
        **kwargs,
    ):
        """Load the key field from multiple save files and display as
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
            Dictionary of arguments for `update_animation`. Matplotlib
            requirement.
        fig_kw : dict
            Dictionary of arguments for arguments for the figure.

        Other Parameters
        ----------------
        All `kwargs` are passed on to `init_animation` and `_ani_save`

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
        >>> animate('rot', dt_equations=0.1, dt_frame_in_sec=50, clim=(-5, 5))
        >>> animate('rot', clim=(-300, 300), fig_kw={"figsize": (14, 4)})
        >>> animate('rot', tmax=25, clim=(-5, 5), save_file='True')
        >>> animate('rot', clim=(-5, 5), save_file='~/fluidsim.gif', codec='imagemagick')

        .. TODO: Use FuncAnimation with blit=True option.

        Notes
        -----

        This method is kept as generic as possible. Any arguments specific to
        1D, 2D or 3D animations or specific to a type of output are be passed
        via keyword arguments (``kwargs``) into its respective
        ``init_animation`` or ``update_animation`` methods.

        """
        if mpi.rank > 0:
            raise NotImplementedError("Do NOT use this function with MPI !")

        self.init_animation(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
        )

        if isinstance(repeat, int):
            nb_repeat = repeat
            repeat = False
        else:
            nb_repeat = 1

        self._animation = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            nb_repeat * len(self.ani_times),
            fargs=fargs.items(),
            interval=dt_frame_in_sec * 1000,
            blit=False,
            repeat=repeat,
        )

        if save_file:
            if not isinstance(save_file, str):
                save_file = r"~/fluidsim_movie.mp4"

            self._ani_save(save_file, dt_frame_in_sec, **kwargs)

    def interact(
        self,
        key_field=None,
        dt_frame_in_sec=0.3,
        dt_equations=None,
        tmin=None,
        tmax=None,
        fig_kw={},
        **kwargs,
    ):
        """Launches an interactive plot.

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
        fig_kw : dict
            Dictionary of arguments for arguments for the figure.

        Other Parameters
        ----------------
        All `kwargs` are passed on to `init_animation` and `_ani_save`

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

        Notes
        -----
        Installation instructions for notebook::

          pip install ipywidgets
          jupyter nbextension enable --py widgetsnbextension

        Restart the notebook and call the function using::

          >>> %matplotlib notebook

        For JupyterLab::

          pip install ipywidgets ipympl
          jupyter labextension install @jupyter-widgets/jupyterlab-manager

        Restart JupyterLab and call the function using::

          >>> %matplotlib widget

        """
        try:
            from ipywidgets import interact, widgets
        except ImportError:
            raise ImportError(
                "See fluidsim.base.output.movies.interact docstring."
            )

        if not is_run_from_jupyter():
            raise ValueError("Works only inside Jupyter.")

        self.init_animation(
            key_field, 0, dt_equations, tmin, tmax, fig_kw, **kwargs
        )
        if tmin is None:
            tmin = self.ani_times[0]
        if tmax is None:
            tmax = self.ani_times[-1]
        if dt_equations is None:
            dt_equations = self.ani_times[1] - self.ani_times[0]

        slider = widgets.FloatSlider(
            min=float(tmin),
            max=float(tmax),
            step=float(dt_equations),
            value=float(tmin),
        )

        def widget_update(time):
            frame = np.argmin(abs(self.ani_times - time))
            self.update_animation(frame)
            self.fig.canvas.draw()

        interact(widget_update, time=slider)

    def _ani_save(self, path_file, dt_frame_in_sec, codec="ffmpeg", **kwargs):
        """Saves the animation using `matplotlib.animation.writers`."""

        path_file = os.path.expandvars(path_file)
        path_file = os.path.expanduser(path_file)
        avail = animation.writers.list()
        if len(avail) == 0:
            raise ValueError(
                "Please install a codec library. For e.g. ffmpeg, mencoder, "
                "imagemagick, html"
            )

        elif codec not in avail:
            print("Using one of the available codecs: {}".format(avail.keys()))
            codec = list(avail.keys())[0]

        Writer = animation.writers[codec]

        print("Saving movie to ", path_file, "...")
        writer = Writer(
            fps=1.0 / dt_frame_in_sec, metadata=dict(artist="FluidSim")
        )
        # _animation is a FuncAnimation object
        self._animation.save(path_file, writer=writer, dpi=150)


class MoviesBase1D(MoviesBase):
    """Base class defining most generic functions for movies for 1D data."""

    def init_animation(
        self, key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
    ):
        """Initializes animated figure."""

        super().init_animation(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
        )

        ax = self.ax
        (self._ani_line,) = ax.plot([], [])

        if "xmax" in kwargs:
            ax.set_xlim(0, kwargs["xmax"])
        else:
            ax.set_xlim(0, self.output.sim.oper.lx)

        if "ymax" in kwargs:
            ax.set_ylim(1e-16, kwargs["ymax"])

    def update_animation(self, frame, **fargs):
        """Loads contour data and updates figure."""
        print("update_animation for frame", frame, "       ", end="\r")
        time = self.ani_times[frame % len(self.ani_times)]
        get_field_to_plot = self.phys_fields.get_field_to_plot
        y, time = get_field_to_plot(time=time, key=self.key_field)
        x = self._get_axis_data()

        self._ani_line.set_data(x, y)
        self.ax.set_title(
            self.key_field + f", $t = {time:.3f}$\n" + self.output.summary_simul
        )

        return self._ani_line

    def _get_axis_data(self):
        """Get axis data.

        Returns
        -------

        x : array
          x-axis data.

        """
        try:
            x = self.oper.xs
        except AttributeError:
            x = self.oper.x_seq
        return x


class MoviesBase2D(MoviesBase):
    """Base class defining most generic functions for movies for 2D data."""

    def _get_axis_data(self):
        """Get axis data.

        Returns
        -------

        x : array
          x-axis data.

        y : array
          y-axis data.

        """

        if not hasattr(self, "_equation"):
            equation = None
        else:
            equation = self._equation

        if (
            equation is None
            or equation.startswith("iz=")
            or equation.startswith("z=")
        ):
            x = self.oper.get_grid1d_seq("x")
            y = self.oper.get_grid1d_seq("y")
        elif equation.startswith("iy=") or equation.startswith("y="):
            x = self.oper.get_grid1d_seq("x")
            y = self.oper.get_grid1d_seq("z")
        elif equation.startswith("ix=") or equation.startswith("x="):
            x = self.oper.get_grid1d_seq("y")
            y = self.oper.get_grid1d_seq("z")
        else:
            raise NotImplementedError

        return x, y
