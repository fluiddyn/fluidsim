"""Operators 0d (:mod:`fluidsim.operators.operators0d`)
=======================================================

Provides

.. autoclass:: Operators0D
   :members:
   :private-members:

"""

from __future__ import division

from builtins import object


class Operators0D(object):
    """0D operators.

    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        params._set_child('oper')

    def __init__(self, params=None, SEQUENTIAL=None):

        self.params = params
        self.shapeX_seq = self.shapeX_loc = []

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""

        return ''

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        return '0d simulation\n'

    def gather_Xspace(self, a):
        """Gather an array (mpi), in this case, just return the array."""
        return a
