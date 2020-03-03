import unittest

from fluidsim.operators.operators3d import OperatorsPseudoSpectral3D

from .test_operators2d import TestCoarse as _TestCoarse


class TestCoarse(_TestCoarse):
    Oper = OperatorsPseudoSpectral3D
    nb_dim = 3


del _TestCoarse

if __name__ == "__main__":
    unittest.main()
