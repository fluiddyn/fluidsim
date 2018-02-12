"""Physical fields output (:mod:`fluidsim.solvers.ns2d.strat.output.phys_fields`)
================================================================================

Provides:

.. autoclass:: PhysFields2DStrat
   :members:
   :private-members:

"""
from __future__ import print_function

from fluidsim.base.output.phys_fields import PhysFieldsBase2D


class PhysFields2DStrat(PhysFieldsBase2D):
    """Class physical fields of solver ns2d.strat """

    def _ani_update(self, frame, **fargs):

        super(PhysFields2DStrat, self)._ani_update(frame, **fargs)

        if 'ratio_omegas' in self.__dict__:
            title = (self._ani_key +
                     ', R = {0:.2f}'.format(
                         self.output.ratio_omegas) +
                     ', F = {0:.1f}'.format(
                         self.output.froude_number) +
                     ', $t = {0:.3f}$'.format(self._ani_t[frame]))
        else:
            title = (self._ani_key + ', $t = {0:.3f}$'.format(
                self._ani_t[frame]))

        self._ani_ax.set_title(title)
