import pytest
import numpy as np

from fluidsim.operators.coord_system3d import CoordSystem3DConverter

shape = (4, 4, 4)


@pytest.fixture(scope="module")
def converter():
    # TODO: fix this
    x = np.zeros(shape)
    y = np.zeros(shape)
    z = np.zeros(shape)

    return CoordSystem3DConverter(x, y, z)


# TODO: add more values for vector_kind
@pytest.mark.parametrize("vector_kind", ["pure-radial", "pure-rh"])
def test_coord_system_converter(vector_kind, converter, allclose):
    match vector_kind:
        case "pure-radial":
            # TODO: fix this
            vx = np.zeros(shape)
            vy = np.zeros(shape)
            vz = np.zeros(shape)
        case "pure-rh":
            # TODO: fix this
            vx = np.zeros(shape)
            vy = np.zeros(shape)
            vz = np.zeros(shape)
        case _:
            raise ValueError

    # TODO: call all the methods
    # example
    vh, vt, vz = converter.compute_cylindrical_components(vx, vy, vz)

    # TODO: check the results with allclose
    match vector_kind:
        case "pure-radial":
            # TODO: fix this
            # bad example
            assert allclose(vx, vy)
        case "pure-rh":
            # TODO: fix this
            pass
        case _:
            raise ValueError


def test_compute_r_theta(converter, allclose):
    r_theta = converter.compute_r_theta()
    # TODO: fix this
    assert allclose(r_theta, r_theta)
