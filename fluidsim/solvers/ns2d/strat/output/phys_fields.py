"""Physical fields output (:mod:`fluidsim.solvers.ns2d.strat.output.phys_fields`)
================================================================================

Provides:

.. autoclass:: PhysFields2DStrat
   :members:
   :private-members:

"""
from __future__ import print_function

from fluidsim.base.output.phys_fields2d import PhysFieldsBase2D


class PhysFields2DStrat(PhysFieldsBase2D):
    """Class physical fields of solver ns2d.strat """

    def update_animation(self, frame, **fargs):

        super(PhysFields2DStrat, self).update_animation(frame, **fargs)

        if "ratio_omegas" in self.__dict__:
            title = (
                self.key_field
                + ", R = {0:.2f}".format(self.output.ratio_omegas)
                + ", F = {0:.1f}".format(self.output.froude_number)
                + ", $t = {0:.3f}$".format(self.ani_times[frame])
            )
        else:
            title = (
                self.key_field + ", $t = {0:.3f}$".format(self.ani_times[frame])
            )

        self.ax.set_title(title)
