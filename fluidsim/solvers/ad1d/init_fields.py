"""Initialization of the field (:mod:`fluidsim.solvers.ad1d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsAD1D
   :members:

.. autoclass:: InitFieldsCos
   :members:

.. autoclass:: InitFieldsGaussian
   :members:

"""
from __future__ import division
from past.utils import old_div
import numpy as np

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsCos(SpecificInitFields):
    tag = 'cos'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsCos, cls)._complete_params_with_default(params)
        # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        oper = self.sim.oper
        s = np.cos(2*np.pi * oper.xs / oper.Lx)
        self.sim.state.state_phys[0] = s


class InitFieldsGaussian(SpecificInitFields):
    tag = 'gaussian'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsGaussian, cls)._complete_params_with_default(params)
        # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        oper = self.sim.oper
        s = np.exp(-(10*(oper.xs - old_div(oper.Lx,2)))**2)
        self.sim.state.state_phys[0] = s


class InitFieldsAD1D(InitFieldsBase):
    """Init the fields for the solver AD1D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[InitFieldsCos, InitFieldsGaussian])
