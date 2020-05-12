"""Base classes for Operators  (:mod:`fluidsim.operators.base`)
===============================================================

Numerical method agnostic base operator classes

Provides:

.. autoclass:: OperatorBase1D
   :members:
   :private-members:

"""

import numpy as np


class OperatorBase:
    def _modify_sim_repr_maker(self, sim_repr_maker):
        if not hasattr(self, "produce_str_describing_oper"):
            return

        repr_oper = self.produce_str_describing_oper()

        if (
            self.axes
            and self.axes != ("lat", "lon")
            and self.params.ONLY_COARSE_OPER
        ):
            str_shape, str_volume = repr_oper.split("_")
            ndim = len(self.axes)
            p_oper = self.params.oper
            if ndim == 1:
                shape_reversed = (p_oper.nx,)
            elif ndim == 2:
                shape_reversed = (p_oper.nx, p_oper.ny)
            elif ndim == 3:
                shape_reversed = (p_oper.nx, p_oper.ny, p_oper.nz)
            else:
                raise NotImplementedError
            str_shape = "x".join(str(n) for n in shape_reversed)
            repr_oper = f"{str_shape}_{str_volume}"

        sim_repr_maker.add_word(repr_oper)


class OperatorsBase1D(OperatorBase):
    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {"nx": 48, "Lx": 8.0}
        params._set_child("oper", attribs=attribs)
        return params

    def __init__(self, params=None):
        self.params = params
        self.axes = ("x",)

        self.nx = nx = int(params.oper.nx)
        self.lx = self.Lx = Lx = float(params.oper.Lx)

        self.size = nx
        self.shapeX = self.shapeX_seq = self.shapeX_loc = self.shape = (nx,)
        self.deltax = Lx / nx
        self.x = self.x_seq = self.x_loc = self.xs = self.deltax * np.arange(nx)

    def _str_describing_oper(self):
        if (self.Lx / np.pi).is_integer():
            str_Lx = repr(int(self.Lx / np.pi)) + "pi"
        else:
            str_Lx = f"{self.Lx:.3f}".rstrip("0")

        return str_Lx

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""
        str_Lx = self._str_describing_oper()
        return ("{}_S" + str_Lx).format(self.nx)

    def produce_long_str_describing_oper(self, oper_method="Base"):
        """Produce a string describing the operator."""
        str_Lx = self._str_describing_oper()
        return (
            f"{oper_method} operator 1D,\n"
            + f"nx = {self.nx:6d}\n"
            + "Lx = "
            + str_Lx
            + "\n"
        )
