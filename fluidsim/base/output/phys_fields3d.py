"""Physical fields output 3d
============================

Provides:

.. autoclass:: MoviesBasePhysFields3D
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: PhysFieldsBase3D
   :members:
   :private-members:
   :undoc-members:

"""

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import mpi

from .phys_fields2d import MoviesBasePhysFields2D, PhysFieldsBase2D


def _get_xylabels_from_equation(equation):
    if equation.startswith("iz=") or equation.startswith("z="):
        xlabel = "x"
        ylabel = "y"
    elif equation.startswith("iy=") or equation.startswith("y="):
        xlabel = "x"
        ylabel = "z"
    elif equation.startswith("ix=") or equation.startswith("x="):
        xlabel = "y"
        ylabel = "z"
    else:
        raise NotImplementedError
    return xlabel, ylabel


class MoviesBasePhysFields3D(MoviesBasePhysFields2D):
    def _init_labels(self, xlabel=None, ylabel=None):
        """Initialize the labels."""
        if xlabel is None or ylabel is None:
            _xlabel, _ylabel = _get_xylabels_from_equation(
                self.phys_fields._equation
            )
        if xlabel is None:
            xlabel = _xlabel
        if ylabel is None:
            ylabel = _ylabel
        self.ax.set_xlabel(xlabel, fontdict=self.font)
        self.ax.set_ylabel(ylabel, fontdict=self.font)


class PhysFieldsBase3D(PhysFieldsBase2D):
    _cls_movies = MoviesBasePhysFields3D

    def __init__(self, output):
        super().__init__(output)
        self.set_equation_crosssection("iz=0")

    def set_equation_crosssection(self, equation):
        """Set the equation defining the cross-section.

        Parameters
        ----------

        equation : str

          The equation can be of the shape 'iz=2', 'z=1', ...

        """
        self._equation = equation
        if equation.startswith("iz=") or equation.startswith("z="):
            self.key_vec_xaxis = "vx"
            self.key_vec_yaxis = "vy"
        elif equation.startswith("iy=") or equation.startswith("y="):
            self.key_vec_xaxis = "vx"
            self.key_vec_yaxis = "vz"
        elif equation.startswith("ix=") or equation.startswith("x="):
            self.key_vec_xaxis = "vy"
            self.key_vec_yaxis = "vz"
        else:
            raise NotImplementedError

    def plot(
        self,
        field=None,
        time=None,
        QUIVER=True,
        vector="v",
        equation=None,
        nb_contours=20,
        type_plot="contourf",
        vmin=None,
        vmax=None,
        cmap=None,
        numfig=None,
        SCALED=True,
    ):
        """Plot a field.

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

        if equation is not None:
            self.set_equation_crosssection(equation)
        equation = self._equation

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

        xlabel, ylabel = _get_xylabels_from_equation(equation)
        vecx = vector + xlabel
        vecy = vector + ylabel

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
        else:
            # we have to get the field from a file
            self.set_of_phys_files.update_times()
            if key_field not in self.sim.state.keys_state_phys:
                error_message = (
                    f'key "{key_field}" not in state.keys_state_phys '
                    f"({self.sim.state.keys_state_phys})."
                )

                if time is not None:
                    error_message += (
                        "\nThe quantity cannot be computed because "
                        "time is not None."
                    )
                elif key_field in self.sim.state.keys_computable:
                    if self.sim.params.ONLY_COARSE_OPER:
                        error_message += (
                            f'\n"{key_field}" in sim.state.keys_computable '
                            "but sim.params.ONLY_COARSE_OPER is True"
                        )
                else:
                    error_message += (
                        f"\nThe quantity cannot be computed because "
                        '"{key_field}" in sim.state.keys_computable.'
                    )

                raise ValueError(error_message)

        if not is_field_ready:
            field, time = self.get_field_to_plot(
                key=key_field, time=time, equation=equation
            )
            if QUIVER:
                vecx, time = self.get_field_to_plot(
                    key=vecx, time=time, equation=equation
                )
                vecy, time = self.get_field_to_plot(
                    key=vecy, time=time, equation=equation
                )

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe()
            else:
                fig, ax = self.output.figure_axe(numfig=numfig)

            x_seq, y_seq = self._get_axis_data(equation)

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
            quiver, vmax = self._quiver_plot(ax, vecx, vecy)
        else:
            vmax = None

        if mpi.rank == 0:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            self._set_title(ax, key_field, time, vmax)

            if SCALED:
                ax.axis("scaled")

            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(1e-3)

    def plot_mean(
        self,
        field=None,
        tmin=None,
        tmax=None,
        QUIVER=True,
        vector="v",
        equation=None,
        nb_contours=20,
        type_plot="contourf",
        vmin=None,
        vmax=None,
        cmap=None,
        numfig=None,
        SCALED=True,
    ):
        """Plot the time average of a field.

        Parameters
        ----------

        field : str, optional

        tmin : number, optional

        tmax : number, optional

        QUIVER : True

        vecx : 'ux'

        vecy : 'uy'

        nb_contours : 20

        type_plot : 'contourf'

        vmin : None

        vmax : None

        cmap : None (usually 'viridis')

        numfig : None

        SCALED : True

        """

        if equation is not None:
            self.set_equation_crosssection(equation)
        equation = self._equation

        key_field = None
        if field is None:
            key_field = self.get_key_field_to_plot(forbid_compute=True)
        elif isinstance(field, str):
            key_field = field

        assert key_field is not None

        xlabel, ylabel = _get_xylabels_from_equation(equation)
        vecx = vector + xlabel
        vecy = vector + ylabel

        keys_state_phys = self.sim.state.keys_state_phys
        if vecx not in keys_state_phys or vecy not in keys_state_phys:
            QUIVER = False

        self.set_of_phys_files.update_times()
        times = self.set_of_phys_files.times
        if tmin is None:
            # get tmin from times
            tmin = times.min()

        if tmax is None:
            # get tmax from times
            tmax = times.max()

        # we have to get the field from a file
        if key_field not in self.sim.state.keys_state_phys:
            error_message = (
                f'key "{key_field}" not in state.keys_state_phys '
                f"({self.sim.state.keys_state_phys})."
            )
            raise ValueError(error_message)

        # get times for average
        times_avg = times[(times >= tmin) & (times <= tmax)]

        # initialize
        time = times_avg[0]
        field_onetime, time = self.get_field_to_plot(
            key=key_field, time=time, equation=equation
        )
        field_avg = np.zeros_like(field_onetime)
        vecx_avg = np.zeros_like(field_onetime)
        vecy_avg = np.zeros_like(field_onetime)
        # time average
        for time in times_avg:
            field_onetime, time = self.get_field_to_plot(
                key=key_field, time=time, equation=equation
            )
            field_avg += field_onetime
            if QUIVER:
                field_onetime, time = self.get_field_to_plot(
                    key=vecx, time=time, equation=equation
                )
                vecx_avg += field_onetime
                field_onetime, time = self.get_field_to_plot(
                    key=vecy, time=time, equation=equation
                )
                vecy_avg += field_onetime
        field_avg /= times_avg.size
        vecx_avg /= times_avg.size
        vecy_avg /= times_avg.size

        # plot
        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe()
            else:
                fig, ax = self.output.figure_axe(numfig=numfig)

            x_seq, y_seq = self._get_axis_data(equation)

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
                    field_avg,
                    nb_contours,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                fig.colorbar(contours)
                fig.contours = contours
            elif type_plot == "pcolor":
                pc = ax.pcolormesh(
                    x_seq,
                    y_seq,
                    field_avg,
                    shading="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                fig.colorbar(pc)
        else:
            ax = None

        if QUIVER:
            quiver, vmax = self._quiver_plot(ax, vecx_avg, vecy_avg)
        else:
            vmax = None

        if mpi.rank == 0:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            title = key_field + f" $(tmin={tmin:8.6g}, tmax={tmax:8.6g})$"
            if vmax is not None:
                title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
            ax.set_title(title + "\n" + self.output.summary_simul)

            if SCALED:
                ax.axis("scaled")

            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(1e-3)
