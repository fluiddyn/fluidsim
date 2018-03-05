"""Base classes for Operators  (:mod:`fluidsim.operators.base`)
===============================================================

Numerical method agnostic base operator classes
 
Provides:

.. autoclass:: OperatorBase1D
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np


class OperatorsBase1D(object):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {'nx': 48, 'Lx': 8.}
        params._set_child('oper', attribs=attribs)
        return params

    def __init__(self, params=None):
        """
        .. todo::

            Cleanup variable names to conform with fluidfft variables.
            For eg.,

             - self.xs -> self.x
             - self.shape -> self.shapeX

        """
        self.params = params

        self.nx = nx = int(params.oper.nx)
        self.Lx = Lx = float(params.oper.Lx)

        self.size = nx
        self.shapeX = self.shapeX_seq = self.shapeX_loc = self.shape = (nx,)
        self.deltax = dx = Lx / nx
        self.x = self.x_seq = self.x_loc = self.xs = np.linspace(0, Lx, nx)

    def _str_describing_oper(self):
        if (self.Lx / np.pi).is_integer():
            str_Lx = repr(int(self.Lx / np.pi)) + 'pi'
        else:
            str_Lx = '{:.3f}'.format(self.Lx).rstrip('0')

        return str_Lx

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""
        str_Lx = self._str_describing_oper()
        return ('{}_S' + str_Lx).format(self.nx)

    def produce_long_str_describing_oper(self, oper_method='Base'):
        """Produce a string describing the operator."""
        str_Lx = self._str_describing_oper()
        return (
            '{} operator 1D,\n'.format(oper_method) +
            'nx = {0:6d}\n'.format(self.nx) +
            'Lx = ' + str_Lx + '\n')
