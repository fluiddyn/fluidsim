"""Operators 0d (:mod:`fluidsim.operators.operators0d`)
=======================================================

Provides

.. autoclass:: Operators0D
   :members:
   :private-members:

"""

from fluiddyn.util import mpi


class Operators0D:
    """0D operators."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        params._set_child("oper")

    def __init__(self, params=None, SEQUENTIAL=None):
        if mpi.nb_proc > 1:
            raise ValueError

        self.params = params
        self.axes = tuple()
        self.shapeX_seq = self.shapeX_loc = []

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""

        return ""

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        return "0d simulation\n"

    def gather_Xspace(self, a):
        """Gather an array (mpi), in this case, just return the array."""
        return a
