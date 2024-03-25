from warnings import warn
from ..output.phys_fields2d import PhysFieldsBase2D, MoviesBasePhysFields2D


class Movies2DSphericalHarmo(MoviesBasePhysFields2D):
    """Base class defining most functions for movies for 2D spherical data."""

    def _get_axis_data(self, shape=None):
        """Get axis data.

        Returns
        -------

        lons : array
          x-axis data.

        lats : array
          y-axis data.

        """
        if shape:
            warn("_get_axis_data: shape parameter ignored.")

        x = self._get_grid1d_seq("lon")
        y = self._get_grid1d_seq("lat")

        return x, y


class PhysFieldsSphericalHarmo(PhysFieldsBase2D):
    _cls_movies = Movies2DSphericalHarmo

    def _set_title(self, ax, key, time, vmax=None):
        title = (
            key
            + f", $t = {time:.3f}$, "
            + self.output.name_solver
            + rf", $l_\max = {self.params.oper.lmax:d}$"
        )
        if vmax is not None:
            title += r", $|\vec{v}|_{max} = $" + f"{vmax:.3f}"
        ax.set_title(title)

    def _compute_skip_quiver(self):
        # 4% of the Lx it is a great separation between vector arrows.
        lx = max(self.oper.lons)
        nx = len(self.oper.lons)
        delta_quiver = 0.04 * lx
        skip = (nx / lx) * delta_quiver
        skip = int(round(skip))
        if skip < 1:
            skip = 1
        return skip
