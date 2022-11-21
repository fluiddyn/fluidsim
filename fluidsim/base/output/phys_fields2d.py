"""Physical fields output 2d
============================

Provides:

.. autoclass:: MoviesBasePhysFields2D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: PhysFieldsBase2D
   :members:
   :private-members:
   :undoc-members:

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from fluiddyn.util import mpi
from fluidsim_core.output.movies import MoviesBasePhysFields
from ..params import Parameters

from .phys_fields import PhysFieldsBase


class MoviesBasePhysFields2D(MoviesBasePhysFields):
    """Methods required to animate physical fields HDF5 files."""

    def _init_fig(self, field, time, vec_xaxis=None, vec_yaxis=None, **kwargs):
        """Initialize only the figure and related matplotlib objects. This
        method is shared by both ``animate`` and ``online_plot``
        functionalities.

        """
        super()._init_fig(field, time, vec_xaxis, vec_yaxis, **kwargs)

        INSET = True if "INSET" not in kwargs else kwargs["INSET"]
        try:
            self.output.spatial_means
        except AttributeError:
            INSET = False

        if not self.sim.time_stepping.is_simul_completed():
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

    def update_animation(self, frame, **fargs):
        """Loads data and updates figure."""
        super().update_animation(frame, **fargs)

        # INLET ANIMATION
        if self._ANI_INSET:
            time = self.ani_times[frame % len(self.ani_times)]
            times = self._ani_spatial_means_t
            idx_spatial = np.abs(times - time).argmin()
            E = self._ani_spatial_means_key
            self._im_inset[0].set_data(times[:idx_spatial], E[:idx_spatial])

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
    _cls_movies = MoviesBasePhysFields2D

    def _init_skip_quiver(self):
        # 4% of the Lx it is a great separation between vector arrows.
        try:
            delta_quiver = 0.04 * min(self.oper.Lx, self.oper.Ly)
        except AttributeError:
            skip = 1
        else:
            skip = (len(self._get_grid1d_seq("x")) / self.oper.Lx) * delta_quiver
            skip = int(np.round(skip))
            if skip < 1:
                skip = 1
        self._skip_quiver = skip
        return skip

    def set_skip_quiver(self, skip):
        self._skip_quiver = skip

    def get_vector_for_plot(
        self, from_state=False, time=None, interpolate_time=True
    ):
        if from_state:
            vecx, _ = self.get_field_to_plot_from_state(self.key_vec_xaxis)
            vecy, _ = self.get_field_to_plot_from_state(self.key_vec_yaxis)
        else:
            vecx, _ = self.get_field_to_plot(
                key=self.key_vec_xaxis,
                time=time,
                interpolate_time=interpolate_time,
            )
            vecy, _ = self.get_field_to_plot(
                key=self.key_vec_yaxis,
                time=time,
                interpolate_time=interpolate_time,
            )
        return vecx, vecy

    def _set_title(self, ax, key, time, vmax=None):
        title = key + f", $t = {time:.3f}$"
        if vmax is not None:
            title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
        ax.set_title(title + "\n" + self.output.summary_simul)

    def _init_online_plot(self):
        self.key_field = self.params.output.phys_fields.field_to_plot

        field, _ = self.get_field_to_plot_from_state(self.key_field)

        try:
            vec_xaxis, vec_yaxis = self.get_vector_for_plot()
        except ValueError:
            self._can_plot_quiver = False
            vec_xaxis = vec_yaxis = None
        else:
            self._can_plot_quiver = True

        if mpi.rank == 0:
            movies = self.movies
            movies.fig, movies.ax = plt.subplots()
            movies._init_fig(
                field, self.output.sim.time_stepping.t, vec_xaxis, vec_yaxis
            )
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
            if self._can_plot_quiver:
                vec_xaxis, vec_yaxis = self.get_vector_for_plot()

            if mpi.rank == 0:
                # Update figure, quiver and colorbar
                self.movies._im.set_array(field.flatten())
                if self._can_plot_quiver:
                    vmax = np.max(np.sqrt(vec_xaxis**2 + vec_yaxis**2))
                    skip = self._skip_quiver
                    self.movies._ani_quiver.set_UVC(
                        vec_xaxis[::skip, ::skip] / vmax,
                        vec_yaxis[::skip, ::skip] / vmax,
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

            x_seq = self._get_grid1d_seq("x")
            y_seq = self._get_grid1d_seq("y")

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
            elif type_plot is None:
                pass
            else:
                print(
                    f"`{type_plot = }` not implemented. It has to be in "
                    '["contourf", "pcolor", "pcolormesh", None]'
                )
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

            try:
                fig.tight_layout()
            except RuntimeError:
                pass
            fig.canvas.draw()
            plt.pause(1e-3)

    def _quiver_plot(
        self,
        ax,
        vecx="ux",
        vecy="uy",
        XX=None,
        YY=None,
        skip=None,
        normalize_vectors=True,
        **kwargs,
    ):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        if isinstance(vecx, str):
            vecx, time = self.get_field_to_plot(vecx)

        if isinstance(vecy, str):
            vecy, time = self.get_field_to_plot(vecy)

        if XX is None and YY is None:
            x_seq, y_seq = self._get_axis_data(self._equation)
            XX, YY = np.meshgrid(x_seq, y_seq)

        if mpi.rank != 0:
            return None, None

        if skip is None:
            skip = self._skip_quiver
        # copy to avoid a bug
        vecx_c = vecx[::skip, ::skip].copy()
        vecy_c = vecy[::skip, ::skip].copy()

        if normalize_vectors:
            vmax = np.max(np.sqrt(vecx**2 + vecy**2))
            vecx_c /= vmax
            vecy_c /= vmax
        else:
            vmax = None

        quiver = ax.quiver(
            XX[::skip, ::skip],
            YY[::skip, ::skip],
            vecx_c,
            vecy_c,
            **kwargs,
        )

        return quiver, vmax
