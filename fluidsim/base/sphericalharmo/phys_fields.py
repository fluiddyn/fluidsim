

from ..output.phys_fields import PhysFieldsBase2D


class PhysFieldsSphericalHarmo(PhysFieldsBase2D):

    def _set_title(self, ax, key, time, vmax=None):
        title = (key +
                 ', $t = {0:.3f}$, '.format(time) +
                 self.output.name_solver +
                 r', $l_\max = {0:d}$'.format(self.params.oper.lmax))
        if vmax is not None:
            title += r', $|\vec{v}|_{max} = $' + '{0:.3f}'.format(vmax)
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
