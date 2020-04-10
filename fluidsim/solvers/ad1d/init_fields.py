"""Initialization of the field (:mod:`fluidsim.solvers.ad1d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsAD1D
   :members:

.. autoclass:: InitFieldsCos
   :members:

.. autoclass:: InitFieldsGaussian
   :members:

"""
import numpy as np

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsCos(SpecificInitFields):
    tag = "cos"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)

    # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        oper = self.sim.oper
        s = np.cos(2 * np.pi * oper.xs / oper.Lx)
        self.sim.state.state_phys[0] = s


class InitFieldsGaussian(SpecificInitFields):
    tag = "gaussian"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)

    # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        oper = self.sim.oper
        s = np.exp(-((10 * (oper.xs - oper.Lx / 2.0)) ** 2))
        self.sim.state.state_phys[0] = s

        if hasattr(self.sim.state, "statespect_from_statephys"):
            self.sim.state.statespect_from_statephys()


class InitFieldsAD1D(InitFieldsBase):
    """Init the fields for the solver AD1D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[InitFieldsCos, InitFieldsGaussian]
        )
