import pytest
import numpy as np

from fluidsim.operators.coord_system3d import CoordSystem3DConverter

# Use a 3D grid of points in Cartesian coordinates.
# Avoid x=y=0 (the z-axis) to keep cylindrical/spherical angles well-defined.
_n = 4
_x1d = np.linspace(1.0, 2.0, _n)
_y1d = np.linspace(0.5, 1.5, _n)
_z1d = np.linspace(-1.0, 1.0, _n)
_z, _y, _x = np.meshgrid(_z1d, _y1d, _x1d, indexing="ij")
shape = _x.shape


@pytest.fixture(scope="module")
def converter():
    return CoordSystem3DConverter(_x, _y, _z)


@pytest.fixture(scope="module")
def r_h():
    """Horizontal (cylindrical) radius sqrt(x^2 + y^2)."""
    return np.sqrt(_x**2 + _y**2)


@pytest.fixture(scope="module")
def r_sph():
    """Spherical radius sqrt(x^2 + y^2 + z^2)."""
    return np.sqrt(_x**2 + _y**2 + _z**2)


# ---------------------------------------------------------------------------
# compute_r_theta
# ---------------------------------------------------------------------------


def test_compute_r_theta_range(converter):
    """r_theta must lie in [-pi, pi]."""
    r_theta = converter.compute_r_theta()
    assert np.all(r_theta >= -np.pi)
    assert np.all(r_theta <= np.pi)


def test_compute_r_theta_values(converter, allclose):
    """r_theta should equal arctan2(y, x)."""
    r_theta = converter.compute_r_theta()
    expected = np.arctan2(_y, _x)
    assert allclose(r_theta, expected)


def test_compute_r_theta_origin():
    """r_theta must be 0 when x = y = 0 (on the z-axis)."""
    x = np.zeros((3,))
    y = np.zeros((3,))
    z = np.array([1.0, 0.0, -1.0])
    conv = CoordSystem3DConverter(x, y, z)
    r_theta = conv.compute_r_theta()
    assert np.all(r_theta == 0.0)


# ---------------------------------------------------------------------------
# compute_cylindrical_components — pure-radial vector
# ---------------------------------------------------------------------------
# A pure-radial (horizontal) vector points in the direction of (x, y, 0),
# i.e. vx = x/r_h, vy = y/r_h, vz = 0.
# In cylindrical coordinates this should give vh = 1, vt = 0, vz = 0.


@pytest.mark.parametrize(
    "vector_kind",
    ["pure-radial-h", "pure-azimuthal", "pure-vertical", "pure-spherical-radial"],
)
def test_compute_cylindrical_components(
    vector_kind, converter, r_h, r_sph, allclose
):
    match vector_kind:
        case "pure-radial-h":
            # Unit vector in the horizontal radial direction
            vx = _x / r_h
            vy = _y / r_h
            vz = np.zeros(shape)
            vh_exp = np.ones(shape)
            vt_exp = np.zeros(shape)
            vz_exp = np.zeros(shape)

        case "pure-azimuthal":
            # Unit vector in the azimuthal direction: (-y, x, 0) / r_h
            vx = -_y / r_h
            vy = _x / r_h
            vz = np.zeros(shape)
            vh_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vz_exp = np.zeros(shape)

        case "pure-vertical":
            # Unit vector along z
            vx = np.zeros(shape)
            vy = np.zeros(shape)
            vz = np.ones(shape)
            vh_exp = np.zeros(shape)
            vt_exp = np.zeros(shape)
            vz_exp = np.ones(shape)

        case "pure-spherical-radial":
            # Unit vector in the spherical radial direction: (x, y, z) / r_sph
            # Cylindrical decomposition: vh = r_h/r_sph, vt = 0, vz = z/r_sph
            vx = _x / r_sph
            vy = _y / r_sph
            vz = _z / r_sph
            vh_exp = r_h / r_sph
            vt_exp = np.zeros(shape)
            vz_exp = _z / r_sph

        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    assert allclose(vh, vh_exp), f"vh mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vz_out, vz_exp), f"vz mismatch for {vector_kind}"


# ---------------------------------------------------------------------------
# compute_cylindrical_components — linearity / inverse
# ---------------------------------------------------------------------------


def test_cylindrical_preserves_norm(converter, allclose):
    """Cylindrical conversion is a rotation: it must preserve the vector norm."""
    rng = np.random.default_rng(0)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    norm2_cart = vx**2 + vy**2 + vz**2
    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    norm2_cyl = vh**2 + vt**2 + vz_out**2

    assert allclose(norm2_cyl, norm2_cart)


# ---------------------------------------------------------------------------
# compute_radial_component
# ---------------------------------------------------------------------------


def test_compute_radial_component_pure_radial(converter, r_sph, allclose):
    """A pure horizontal-radial unit vector should have radial component 1."""
    vx = _x / r_sph
    vy = _y / r_sph
    vz = _z / r_sph
    vr = converter.compute_radial_component(vx, vy, vz)
    assert allclose(vr, np.ones(shape))


def test_compute_radial_component_pure_azimuthal(converter, r_h, allclose):
    """A pure azimuthal unit vector is perpendicular to r_h → radial component 0."""
    vx = -_y / r_h
    vy = _x / r_h
    vz = np.zeros(shape)
    vr = converter.compute_radial_component(vx, vy, vz)
    assert allclose(vr, np.zeros(shape))


# ---------------------------------------------------------------------------
# compute_spherical_components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector_kind",
    ["pure-spherical-radial", "pure-azimuthal", "pure-polar"],
)
def test_compute_spherical_components(
    vector_kind, converter, r_h, r_sph, allclose
):
    match vector_kind:
        case "pure-spherical-radial":
            # Unit vector along spherical r: (x, y, z)/r_sph
            vx = _x / r_sph
            vy = _y / r_sph
            vz = _z / r_sph
            vr_exp = np.ones(shape)
            vt_exp = np.zeros(shape)  # azimuthal
            vp_exp = np.zeros(shape)  # polar

        case "pure-azimuthal":
            # Unit vector along azimuthal phi: (-y, x, 0)/r_h
            vx = -_y / r_h
            vy = _x / r_h
            vz = np.zeros(shape)
            vr_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vp_exp = np.zeros(shape)

        case "pure-polar":
            # Unit vector along polar theta (e_theta): (x*z, y*z, -r_h^2) / (r_sph * r_h)
            vx = _x * _z / (r_sph * r_h)
            vy = _y * _z / (r_sph * r_h)
            vz = -(r_h**2) / (r_sph * r_h)
            vr_exp = np.zeros(shape)
            vt_exp = np.zeros(shape)
            vp_exp = np.ones(shape)

        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    assert allclose(vr, vr_exp), f"vr mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vp, vp_exp), f"vp mismatch for {vector_kind}"


def test_spherical_preserves_norm(converter, allclose):
    """Spherical conversion is a rotation: it must preserve the vector norm."""
    rng = np.random.default_rng(1)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    norm2_cart = vx**2 + vy**2 + vz**2
    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    norm2_sph = vr**2 + vt**2 + vp**2

    assert allclose(norm2_sph, norm2_cart)


def test_spherical_radial_equals_radial_component(converter, allclose):
    """The spherical vr component must equal compute_radial_component."""
    rng = np.random.default_rng(2)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    vr_sph, _, _ = converter.compute_spherical_components(vx, vy, vz)
    vr_direct = converter.compute_radial_component(vx, vy, vz)

    assert allclose(vr_sph, vr_direct)
