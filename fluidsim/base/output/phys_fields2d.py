"""Physical fields output 2d (:mod:`fluidsim.base.output.phys_fields2d`)
========================================================================

Provides:

.. autoclass:: MoviesBasePhysFields2D
   :members:
   :private-members:

.. autoclass:: PhysFieldsBase2D
   :members:
   :private-members:

"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from fluiddyn.util import mpi
from .movies import MoviesBase2D
from ..params import Parameters

from .phys_fields import PhysFieldsBase


class MoviesBasePhysFields2D(MoviesBase2D):
    """Methods required to animate physical fields HDF5 files."""

    def __init__(self, output, phys_fields):
        self.phys_fields = phys_fields
        super().__init__(output)
        self._equation = None

    def init_animation(
        self, key_field, numfig, dt_equations, tmin, tmax, fig_kw, **kwargs
    ):
        """Initialize list of files and times, pcolor plot, quiver and colorbar."""
        self.phys_fields.set_of_phys_files.update_times()
        self.time_files = self.phys_fields.set_of_phys_files.times

        if dt_equations is None:
            dt_equations = np.median(np.diff(self.time_files))
            print(f"dt_equations = {dt_equations:.4f}")

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

        self._has_uxuy = self.sim.state.has_vars("ux", "uy")

        get_field_to_plot = self.phys_fields.get_field_to_plot
        field, time = get_field_to_plot(self.key_field)
        if self._has_uxuy:
            ux, time = get_field_to_plot("ux")
            uy, time = get_field_to_plot("uy")
        else:
            ux = uy = None

        INSET = True if "INSET" not in kwargs else kwargs["INSET"]
        self._init_fig(field, ux, uy, INSET, **kwargs)

        self._clim = kwargs.get("clim")
        self._set_clim()

    def _init_fig(self, field, ux=None, uy=None, INSET=True, **kwargs):
        """Initialize only the figure and related matplotlib objects. This
        method is shared by both ``animate`` and ``online_plot``
        functionalities.

        """
        self._step = step = 1 if "step" not in kwargs else kwargs["step"]
        self._QUIVER = True if "QUIVER" not in kwargs else kwargs["QUIVER"]

        x, y = self._get_axis_data(shape=field.shape)
        XX, YY = np.meshgrid(x[::step], y[::step])
        field = field[::step, ::step]

        self._im = self.ax.pcolormesh(XX, YY, field, shading="nearest")
        self._ani_cbar = self.fig.colorbar(self._im)

        self._has_uxuy = self.sim.state.has_vars("ux", "uy")

        if self._has_uxuy and self._QUIVER:
            self._ani_quiver, vmax = self.phys_fields._quiver_plot(
                self.ax, ux, uy, XX, YY
            )

        try:
            self.output.spatial_means
        except AttributeError:
            INSET = False

        if INSET and not self.sim.time_stepping.is_simul_completed():
            INSET = False

        self._ANI_INSET = INSET
        if self._ANI_INSET:
            try:
                (
                    self._ani_spatial_means_t,
                    self._ani_spatial_means_key,
                ) = self._get_spatial_means()
            except FileNotFoundError:
                print("No spatial means file => no inset plot.")
                self._ANI_INSET = False
                return

            left, bottom, width, height = [0.53, 0.67, 0.2, 0.2]
            ax2 = self.fig.add_axes([left, bottom, width, height])

            ax2.set_xlabel("t", labelpad=0.1)
            ax2.set_ylabel("E", labelpad=0.1)

            # Format of the ticks in ylabel
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

            ax2.set_xlim(0, self._ani_spatial_means_t.max())
            # Correct visualization inset_animation 10% of the difference
            # value_max-value-min
            ax2.set_ylim(
                self._ani_spatial_means_key.min(),
                self._ani_spatial_means_key.max()
                + (
                    0.1
                    * abs(
                        self._ani_spatial_means_key.min()
                        - self._ani_spatial_means_key.max()
                    )
                ),
            )

            ax2.plot(
                self._ani_spatial_means_t,
                self._ani_spatial_means_key,
                linewidth=0.8,
                color="grey",
                alpha=0.4,
            )
            self._im_inset = ax2.plot([0], [0], color="red")

    def _get_axis_data(self, shape=None):
        """Get 1D arrays for setting the axes."""

        x, y = super()._get_axis_data()
        if shape is not None and (y.shape[0], x.shape[0]) != shape:
            path_file = os.path.join(self.output.path_run, "params_simul.xml")
            params = Parameters(path_file=path_file)
            x = np.arange(0, params.oper.Lx, params.oper.nx)
            y = np.arange(0, params.oper.Ly, params.oper.ny)

        return x, y

    def update_animation(self, frame, **fargs):
        """Loads data and updates figure."""

        print("update_animation for frame", frame, "       ", end="\r")
        time = self.ani_times[frame % len(self.ani_times)]
        step = self._step
        get_field_to_plot = self.phys_fields.get_field_to_plot

        field, time = get_field_to_plot(
            time=time,
            key=self.key_field,
            equation=self._equation,
            interpolate_time=True,
        )

        field = field[::step, ::step]

        # Update figure, quiver and colorbar
        self._im.set_array(field.flatten())

        if self._has_uxuy and self._QUIVER:
            ux, time = get_field_to_plot(
                time=time,
                key="ux",
                equation=self._equation,
                interpolate_time=True,
            )
            uy, time = get_field_to_plot(
                time=time,
                key="uy",
                equation=self._equation,
                interpolate_time=True,
            )
            vmax = np.max(np.sqrt(ux**2 + uy**2))
            skip = self.phys_fields._skip_quiver
            self._ani_quiver.set_UVC(
                ux[::skip, ::skip] / vmax, uy[::skip, ::skip] / vmax
            )
        else:
            vmax = None

        self._im.autoscale()
        self._set_clim()

        # INLET ANIMATION
        if self._ANI_INSET:
            idx_spatial = np.abs(self._ani_spatial_means_t - time).argmin()
            t = self._ani_spatial_means_t
            E = self._ani_spatial_means_key

            self._im_inset[0].set_data(t[:idx_spatial], E[:idx_spatial])

        self.phys_fields._set_title(self.ax, self.key_field, time, vmax)

    def _set_clim(self):
        """Maintains a constant colorbar throughout the animation."""

        clim = self._clim
        if clim is not None:
            self._im.set_clim(*clim)
            ticks = np.linspace(*clim, num=21, endpoint=True)
            self._ani_cbar.set_ticks(ticks)

    def _get_spatial_means(self, key_spatial="E"):
        """Get field for the inset plot."""
        # Check if key_spatial can be loaded.
        keys_spatial = ["E", "EK", "EA"]
        if key_spatial not in keys_spatial:
            raise ValueError("key_spatial not in spatial means keys.")

        # Load data for inset plot
        results = self.output.spatial_means.load()
        t = results["t"]
        E = results[key_spatial]

        return t, E


class PhysFieldsBase2D(PhysFieldsBase):
    def _init_skip_quiver(self):
        # 4% of the Lx it is a great separation between vector arrows.
        try:
            delta_quiver = 0.04 * min(self.oper.Lx, self.oper.Ly)
        except AttributeError:
            skip = 1
        else:
            skip = (
                len(self.oper.get_grid1d_seq("x")) / self.oper.Lx
            ) * delta_quiver
            skip = int(np.round(skip))
            if skip < 1:
                skip = 1
        self._skip_quiver = skip
        return skip

    def set_skip_quiver(self, skip):
        self._skip_quiver = skip

    def _init_movies(self):
        self.movies = MoviesBasePhysFields2D(self.output, self)

    def _set_title(self, ax, key, time, vmax=None):
        title = key + f", $t = {time:.3f}$"
        if vmax is not None:
            title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
        ax.set_title(title + "\n" + self.output.summary_simul)

    def _init_online_plot(self):
        self.key_field = self.params.output.phys_fields.field_to_plot

        self._has_uxuy = self.sim.state.has_vars("ux", "uy")

        field, _ = self.get_field_to_plot_from_state(self.key_field)
        if self._has_uxuy:
            ux, _ = self.get_field_to_plot_from_state("ux")
            uy, _ = self.get_field_to_plot_from_state("uy")
        else:
            ux = uy = None

        if mpi.rank == 0:
            movies = self.movies
            movies.fig, movies.ax = plt.subplots()
            movies._init_fig(field, ux, uy)
            movies._im.autoscale()
            movies.fig.canvas.draw()
            plt.pause(1e-6)

    def _online_plot(self):
        """Online plot."""
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_plot >= self.period_plot:
            self.t_last_plot = tsim
            key_field = self.params.output.phys_fields.field_to_plot
            field, _ = self.get_field_to_plot_from_state(key_field)
            if self._has_uxuy:
                ux, _ = self.get_field_to_plot_from_state("ux")
                uy, _ = self.get_field_to_plot_from_state("uy")

            if mpi.rank == 0:
                # Update figure, quiver and colorbar
                self.movies._im.set_array(field.flatten())
                if self._has_uxuy:
                    vmax = np.max(np.sqrt(ux**2 + uy**2))
                    skip = self._skip_quiver
                    self.movies._ani_quiver.set_UVC(
                        ux[::skip, ::skip] / vmax, uy[::skip, ::skip] / vmax
                    )
                else:
                    vmax = None

                self._set_title(self.movies.ax, self.key_field, tsim, vmax)

                self.movies._im.autoscale()
                self.movies.fig.canvas.draw()
                plt.pause(1e-6)

    def plot(
        self,
        field=None,
        time=None,
        QUIVER=True,
        vecx="ux",
        vecy="uy",
        nb_contours=20,
        type_plot="pcolor",
        vmin=None,
        vmax=None,
        cmap=None,
        numfig=None,
        skip_quiver=None,
    ):
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

        type_plot : "pcolor" or "contourf"

        vmin : None

        vmax : None

        cmap : None (usually 'viridis')

        numfig : None

        """

        is_field_ready = False

        self._has_uxuy = self.sim.state.has_vars("ux", "uy")

        key_field = None
        if field is None:
            key_field = self.get_key_field_to_plot(
                forbid_compute=time is not None
            )
        elif isinstance(field, np.ndarray):
            key_field = "given array"
            is_field_ready = True
        elif isinstance(field, str):
            key_field = field

        assert key_field is not None

        keys_state_phys = self.sim.state.keys_state_phys
        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

        if (
            time is None
            and not is_field_ready
            and not self.sim.params.ONLY_COARSE_OPER
        ):
            # we have to get the field from the state
            time = self.sim.time_stepping.t
            field, _ = self.get_field_to_plot_from_state(key_field)
            if QUIVER:
                vecx, _ = self.get_field_to_plot_from_state(vecx)
                vecy, _ = self.get_field_to_plot_from_state(vecy)
        else:
            # we have to get the field from a file
            self.set_of_phys_files.update_times()
            tmax = sorted(self.set_of_phys_files.times)[-1]

            if time is None or time > tmax:
                time = sorted(self.set_of_phys_files.times)[-1]

            if key_field not in self.sim.state.keys_state_phys:
                raise ValueError("key not in state.keys_state_phys")

            field, time = self.get_field_to_plot(
                key=key_field, time=time, interpolate_time=True
            )
            if QUIVER:
                vecx, time = self.get_field_to_plot(
                    key=vecx, time=time, interpolate_time=True
                )
                vecy, time = self.get_field_to_plot(
                    key=vecy, time=time, interpolate_time=True
                )

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe()
            else:
                fig, ax = self.output.figure_axe(numfig=numfig)

            x_seq = self.oper.get_grid1d_seq("x")
            y_seq = self.oper.get_grid1d_seq("y")

            [XX_seq, YY_seq] = np.meshgrid(x_seq, y_seq)
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError:
                print(
                    "Use matplotlib >= 1.5.0 for new standard colorschemes.\
                       Installed matplotlib :"
                    + plt.matplotlib.__version__
                )
                cmap = plt.get_cmap("jet")

            if type_plot == "contourf":
                contours = ax.contourf(
                    x_seq,
                    y_seq,
                    field,
                    nb_contours,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                fig.colorbar(contours)
                fig.contours = contours
            elif type_plot in ["pcolor", "pcolormesh"]:
                pc = ax.pcolormesh(
                    x_seq,
                    y_seq,
                    field,
                    shading="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                fig.colorbar(pc)
            else:
                print(f"`type_plot = {type_plot}` not implemented")
        else:
            ax = None

        if QUIVER:
            quiver, vmax = self._quiver_plot(ax, vecx, vecy, skip=skip_quiver)
        else:
            vmax = None

        if mpi.rank == 0:
            ax.set_xlabel(self.oper.axes[1])
            ax.set_ylabel(self.oper.axes[0])
            self._set_title(ax, key_field, time, vmax)

            if self.oper.Lx != self.oper.Ly:
                ax.set_aspect("equal")

            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(1e-3)

    def _quiver_plot(self, ax, vecx="ux", vecy="uy", XX=None, YY=None, skip=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        if isinstance(vecx, str):
            vecx, time = self.get_field_to_plot(vecx)

        if isinstance(vecy, str):
            vecy, time = self.get_field_to_plot(vecy)

        if XX is None and YY is None:
            [XX, YY] = np.meshgrid(
                self.oper.get_grid1d_seq("x"), self.oper.get_grid1d_seq("y")
            )

        if mpi.rank == 0:
            # local variable 'normalize_diff' is assigned to but never used
            # normalize_diff = (
            #     (np.max(np.sqrt(vecx**2 + vecy**2)) -
            #      np.min(np.sqrt(vecx**2 + vecy**2))) /
            #     np.max(np.sqrt(vecx**2 + vecy**2)))
            vmax = np.max(np.sqrt(vecx**2 + vecy**2))
            # Quiver is normalized by the vmax
            # copy to avoid a bug
            if skip is None:
                skip = self._skip_quiver
            vecx_c = vecx[::skip, ::skip].copy()
            vecy_c = vecy[::skip, ::skip].copy()
            quiver = ax.quiver(
                XX[::skip, ::skip],
                YY[::skip, ::skip],
                vecx_c / vmax,
                vecy_c / vmax,
            )
        else:
            quiver = vmax = None

        return quiver, vmax
