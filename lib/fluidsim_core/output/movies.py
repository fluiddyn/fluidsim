"""Movies output
================

Contains base classes which acts as a framework to
implement the method ``animate`` to make movies.

Provides:

.. autoclass:: MoviesBase
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: MoviesBase1D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: MoviesBase2D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: MoviesBasePhysFields
   :members:
   :private-members:
   :undoc-members:

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button
import mpl_toolkits.axes_grid1

from fluiddyn.util import mpi, is_run_from_jupyter

from ..hexa_field import get_edges


class MoviesBase:
    """Base class defining most generic functions for movies."""

    def __init__(self, output):
        params = output.sim.params
        self.output = output
        self.sim = output.sim
        self.params = params.output
        self.oper = self.sim.oper

        self._set_font()

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

    def init_animation(
        self, key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
    ):
        """Initializes animated fig. and list of times of save files to load."""
        self._set_key_field(key_field)
        self._init_ani_times(tmin, tmax, dt_equations)
        self.fig, self.ax = plt.subplots(num=numfig, **fig_kw)
        self._init_labels()

        if not self._interactive:
            return

        # see https://stackoverflow.com/a/44989063
        playerax = self.fig.add_axes([0.05, 0.015, 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)

        self._buttons = []

        def init_button(ax, label, method):
            button = Button(ax, label=label)
            button.on_clicked(method)
            self._buttons.append(button)

        init_button(playerax, "$\u29CF$", self._one_backward)
        init_button(bax, "$\u25C0$", self._backward)
        init_button(sax, "$\u25A0$", self.pause)
        init_button(fax, "$\u25B6$", self._forward)
        init_button(ofax, "$\u29D0$", self._one_forward)

    def resume(self):
        self.paused = False
        self._animation.resume()

    def pause(self, event=None):
        self.paused = True
        self._animation.pause()

    def _forward(self, event=None):
        self._forwards = True
        self.resume()

    def _backward(self, event=None):
        self._forwards = False
        self.resume()

    def _one_forward(self, event=None):
        self._forwards = True
        self.onestep()

    def _one_backward(self, event=None):
        self._forwards = False
        self.onestep()

    def _get_default_tmax(self):
        try:
            return self.sim.time_stepping.t
        except AttributeError:
            return self.phys_fields.set_of_phys_files.get_max_time()

    def _init_ani_times(self, tmin, tmax, dt_equations):
        """Initialization of the variable ani_times for one animation."""
        self.phys_fields.set_of_phys_files.update_times()
        if tmax is None:
            tmax = self._get_default_tmax()

        if tmin is None:
            tmin = self.phys_fields.set_of_phys_files.get_min_time()

        if tmin > tmax:
            raise ValueError(
                "Error tmin > tmax. Value tmin should be smaller than tmax"
            )

        if dt_equations is None:
            dt_equations = self.params.periods_save.phys_fields

        self.ani_times = np.arange(tmin, tmax, dt_equations)

    def update_animation(self, frame, **fargs):
        """Replace this function to load data for next frame and update the
        figure.

        """
        pass

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
            forbid_compute=True, key_prefered=key_field
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
        interactive=None,
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
        interactive : bool
            Add player buttons (pause, step by step and forward/backward)
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

        self._interactive = interactive
        self.init_animation(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
        )

        if isinstance(repeat, int) and repeat:
            nb_repeat = repeat
            repeat = False
        else:
            nb_repeat = 1

        self._min = 0
        frames = self._max = nb_repeat * len(self.ani_times) - 1
        if interactive:
            frames = self._frames_iterative
            self._forwards = True
            self._index = self._min = 0

        self._animation = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            frames=frames,
            fargs=fargs.items(),
            interval=dt_frame_in_sec * 1000,
            blit=False,
            repeat=repeat,
        )

        if save_file:
            if not isinstance(save_file, str):
                save_file = r"~/fluidsim_movie.mp4"

            self._ani_save(save_file, dt_frame_in_sec, **kwargs)
            return

        self.paused = False
        self.fig.canvas.mpl_connect("button_press_event", self._toggle_pause)

    def _frames_iterative(self):
        while not self.paused:
            self._index += self._forwards - (not self._forwards)
            frame = self._index
            if frame < self._min:
                self._index = frame = self._max
            elif frame > self._max:
                self._index = frame = self._min
            elif frame == self._max or frame == self._min:
                self.pause()
            yield frame

    def onestep(self):
        self.pause()
        if self._index > self._min and self._index < self._max:
            self._index += self._forwards - (not self._forwards)
        elif self._index == self._min:
            if self._forwards:
                self._index += 1
            else:
                self._index = self._max
        elif self._index == self._max:
            if not self._forwards:
                self._index -= 1
            else:
                self._index = self._min

        self.update_animation(self._index)
        self.fig.canvas.draw_idle()

    def _toggle_pause(self, event):
        if event.inaxes != self.fig.axes[0]:
            return
        if self.paused:
            self._animation.resume()
        else:
            self._animation.pause()
        self.paused = not self.paused

    def interact(
        self,
        key_field=None,
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
        except ImportError as exc:
            raise ImportError(
                "See fluidsim_core.output.movies.interact docstring."
            ) from exc

        if not is_run_from_jupyter():
            raise ValueError("Works only inside Jupyter.")

        self._interactive = False
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
        return self.phys_fields._get_axis_data()


class MoviesBasePhysFields(MoviesBase2D):
    def __init__(self, output, phys_fields):
        self.phys_fields = phys_fields
        super().__init__(output)

    def init_animation(
        self, key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
    ):
        """Initialize list of files and times, pcolor plot, quiver and colorbar."""
        self.phys_fields.set_of_phys_files.update_times()
        self.time_files = self.phys_fields.set_of_phys_files.times

        if dt_equations is None:
            dt_equations = np.median(np.diff(self.time_files))
            print(f"{dt_equations = :.4f}")

        if tmax is None:
            tmax = self.time_files.max()

        super().init_animation(
            key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
        )

        dt_file = (self.time_files[-1] - self.time_files[0]) / len(
            self.time_files
        )
        if dt_equations < dt_file / 4:
            raise ValueError("dt_equations < dt_file / 4")

        field, time = self.phys_fields.get_field_to_plot(
            self.key_field, time=tmin
        )

        try:
            vec_xaxis, vec_yaxis = self.phys_fields.get_vector_for_plot(time=time)
        except ValueError:
            self.phys_fields._can_plot_quiver = False
            vec_xaxis = vec_yaxis = None
        else:
            self.phys_fields._can_plot_quiver = True

        self._init_fig(field, vec_xaxis, vec_yaxis, **kwargs)

    def _init_fig(self, field, vec_xaxis=None, vec_yaxis=None, **kwargs):
        """Initialize only the figure and related matplotlib objects. This
        method is shared by both ``animate`` and ``online_plot``
        functionalities.

        """
        self._step = step = 1 if "step" not in kwargs else kwargs["step"]
        self._QUIVER = True if "QUIVER" not in kwargs else kwargs["QUIVER"]

        x, y = self._get_axis_data()
        x, y = x[::step], y[::step]
        XX, YY = np.meshgrid(x, y)
        field = field[::step, ::step]
        assert (len(y), len(x)) == field.shape

        self._im = self.ax.pcolormesh(XX, YY, field, shading="nearest")
        self._ani_cbar = self.fig.colorbar(self._im)

        if self.phys_fields._can_plot_quiver and self._QUIVER:
            skip = self.phys_fields._skip_quiver
            self._ani_quiver, vmax = self.phys_fields._quiver_plot(
                self.ax,
                vec_xaxis[::skip, ::skip],
                vec_yaxis[::skip, ::skip],
                XX,
                YY,
            )

        self._clim = kwargs.get("clim")
        self._set_clim()
        self.phys_fields._set_title(self.ax, self.key_field, time, vmax)

    def update_animation(self, frame, **fargs):
        """Loads data and updates figure."""
        time = self.ani_times[frame % len(self.ani_times)]
        step = self._step

        field, time = self.phys_fields.get_field_to_plot(
            time=time,
            key=self.key_field,
            interpolate_time=True,
        )

        field = field[::step, ::step]

        # Update figure, quiver and colorbar
        self._im.set_array(field.flatten())
        if self.phys_fields._can_plot_quiver and self._QUIVER:
            vec_xaxis, vec_yaxis = self.phys_fields.get_vector_for_plot(time=time)

            vmax = np.max(np.sqrt(vec_xaxis**2 + vec_yaxis**2))
            skip = self.phys_fields._skip_quiver
            self._ani_quiver.set_UVC(
                vec_xaxis[::skip, ::skip] / vmax, vec_yaxis[::skip, ::skip] / vmax
            )
        else:
            vmax = None

        self._im.autoscale()
        self._set_clim()

        self.phys_fields._set_title(self.ax, self.key_field, time, vmax)

    def _set_clim(self):
        """Maintains a constant colorbar throughout the animation."""

        clim = self._clim
        if clim is not None:
            self._im.set_clim(*clim)
            ticks = np.linspace(*clim, num=21, endpoint=True)
            self._ani_cbar.set_ticks(ticks)


class MoviesBasePhysFieldsHexa(MoviesBasePhysFields):
    def _init_fig(self, field, vec_xaxis=None, vec_yaxis=None, **kwargs):
        """Initialize only the figure and related matplotlib objects. This
        method is shared by both ``animate`` and ``online_plot``
        functionalities.

        """
        self._step = step = 1 if "step" not in kwargs else kwargs["step"]
        self._QUIVER = True if "QUIVER" not in kwargs else kwargs["QUIVER"]
        vmin = None if "vmin" not in kwargs else kwargs["vmin"]
        vmax = None if "vmax" not in kwargs else kwargs["vmax"]

        if step != 1:
            raise NotImplementedError

        hexa_x, hexa_y = self._get_axis_data()
        hexa_field = field

        if vmax is None:
            vmax = 0
            for arr in hexa_field.arrays:
                max_elem = arr.max()
                if vmax > max_elem:
                    vmax = max_elem

        if vmin is None:
            vmin = 0
            for arr in hexa_field.arrays:
                min_elem = arr.min()
                if vmin > min_elem:
                    vmin = min_elem

        self._images = []
        for i_elem, arr in enumerate(hexa_field.arrays):

            x_edges = hexa_x.elements[i_elem]["edges"]
            y_edges = hexa_y.elements[i_elem]["edges"]

            im = self.ax.pcolormesh(
                x_edges,
                y_edges,
                arr[0],
                shading="flat",
                vmin=vmin,
                vmax=vmax,
            )

            self._images.append(im)

        self._ani_cbar = self.fig.colorbar(self._images[0])

        hexa_vec_xaxis = vec_xaxis
        hexa_vec_yaxis = vec_yaxis

        xmin, xmax = hexa_x.lims
        ymin, ymax = hexa_y.lims

        percentage_dx_quiver = 4
        dx_quiver = percentage_dx_quiver / 100 * (xmax - xmin)
        nx_quiver = int((xmax - xmin) / dx_quiver)
        ny_quiver = int((ymax - ymin) / dx_quiver)

        self.x_approx_quiver = np.linspace(dx_quiver, xmax - dx_quiver, nx_quiver)
        self.y_approx_quiver = np.linspace(dx_quiver, ymax - dx_quiver, ny_quiver)

        x_quiver = []
        y_quiver = []
        vx_quiver = []
        vy_quiver = []

        self._indices_vectors_in_elems = []

        # assuming 2d...
        iz = 0

        vmax = 0.0

        for i_elem, (vec_xaxis, vec_yaxis) in enumerate(
            zip(hexa_vec_xaxis.arrays, hexa_vec_yaxis.arrays)
        ):

            XX = hexa_x.arrays[i_elem]
            YY = hexa_y.arrays[i_elem]
            x = XX[iz, 0]
            y = YY[iz, :, 0]
            x_edges = get_edges(x)
            y_edges = get_edges(y)

            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()

            vmax_elem = np.max(np.sqrt(vec_xaxis**2 + vec_yaxis**2))
            if vmax_elem > vmax:
                vmax = vmax_elem

            indices_vectors_in_elem = []
            for y_approx in self.y_approx_quiver:
                if y_approx < ymin:
                    continue
                if y_approx > ymax:
                    break
                iy = abs(y - y_approx).argmin()
                for x_approx in self.x_approx_quiver:
                    if x_approx < xmin:
                        continue
                    if x_approx > xmax:
                        break
                    ix = abs(x - x_approx).argmin()

                    x_quiver.append(x[ix])
                    y_quiver.append(y[iy])
                    vx_quiver.append(vec_xaxis[iz, iy, ix])
                    vy_quiver.append(vec_yaxis[iz, iy, ix])
                    indices_vectors_in_elem.append((iz, iy, ix))

            self._indices_vectors_in_elems.append(indices_vectors_in_elem)

        self._ani_quiver = self.ax.quiver(
            x_quiver, y_quiver, vx_quiver / vmax, vy_quiver / vmax
        )

        self._clim = kwargs.get("clim")
        self._set_clim()

        time = hexa_field.time
        vmax = None
        self.phys_fields._set_title(self.ax, self.key_field, time, vmax)

    def update_animation(self, frame, **fargs):
        """Loads data and updates figure."""
        time = self.ani_times[frame % len(self.ani_times)]
        # step = self._step

        hexa_field, time = self.phys_fields.get_field_to_plot(
            time=time,
            key=self.key_field,
            interpolate_time=True,
        )

        iz = 0
        for image, array in zip(self._images, hexa_field.arrays):
            image.set_array(array[iz].flatten())

        hexa_vec_xaxis, hexa_vec_yaxis = self.phys_fields.get_vector_for_plot(
            time=time
        )

        vmax = 0.0

        vx_quiver = []
        vy_quiver = []

        for i_elem, (vec_xaxis, vec_yaxis) in enumerate(
            zip(hexa_vec_xaxis.arrays, hexa_vec_yaxis.arrays)
        ):

            vmax_elem = np.max(np.sqrt(vec_xaxis**2 + vec_yaxis**2))
            if vmax_elem > vmax:
                vmax = vmax_elem
            indices_vectors_in_elem = self._indices_vectors_in_elems[i_elem]
            for (iz, iy, ix) in indices_vectors_in_elem:
                vx_quiver.append(vec_xaxis[iz, iy, ix])
                vy_quiver.append(vec_yaxis[iz, iy, ix])

        self._ani_quiver.set_UVC(vx_quiver / vmax, vy_quiver / vmax)

        self.phys_fields._set_title(self.ax, self.key_field, time, vmax)
