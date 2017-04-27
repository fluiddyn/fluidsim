
import sys
from math import pi

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D as _Operators

from .functions2d_pythran import dealiasing_setofvar
from ..base.setofvariables import SetOfVariables


class OperatorsPseudoSpectral2D(_Operators):

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        if mpi.nb_proc > 1:
            raise NotImplementedError
        else:
            type_fft = 'fft2d.with_fftw2d'

        attribs = {'type_fft': type_fft,
                   'coef_dealiasing': 2./3,
                   'nx': 48,
                   'ny': 48,
                   'Lx': 8,
                   'Ly': 8}
        params._set_child('oper', attribs=attribs)

    def __init__(self, params, SEQUENTIAL=None, goal_to_print=None):

        self.params = params

        super(OperatorsPseudoSpectral2D, self).__init__(
            params.oper.nx, params.oper.ny, params.oper.Lx, params.oper.Ly,
            fft=params.oper.type_fft,
            coef_dealiasing=params.oper.coef_dealiasing)

        self.Lx = self.lx
        self.Ly = self.ly

    def dealiasing(self, *args):
        for thing in args:
            if isinstance(thing, SetOfVariables):
                dealiasing_setofvar(thing, self.where_dealiased,
                                    self.nK0_loc, self.nK1_loc)
            elif isinstance(thing, np.ndarray):
                self.dealiasing_variable(thing)

    def dealiasing_setofvar(self, sov):
        dealiasing_setofvar(sov, self.where_dealiased,
                            self.nK0_loc, self.nK1_loc)
