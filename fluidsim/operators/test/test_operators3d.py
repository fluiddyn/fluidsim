import unittest

from .test_operators2d import TestCoarse as _TestCoarse


class TestCoarse(_TestCoarse):
    nb_dim = 3

    @property
    def Oper(self):
        from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D

        return OperatorsPseudoSpectral3D


del _TestCoarse

if __name__ == "__main__":
    unittest.main()
