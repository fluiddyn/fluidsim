import pytest
import functools
import numpy as np

from fluidsim.operators.coord_system3d import CoordSystem3DConverter

# Use a 3D grid of points in Cartesian coordinates.
# warning: x=y=0 (the z-axis) is particular, cylindrical/spherical not well-defined.
_n = 4
_x1d = np.linspace(0.0, 1.0, _n)
_y1d = np.linspace(0.0, 1.0, _n)
_z1d = np.linspace(-1.0, 1.0, _n)
_z, _y, _x = np.meshgrid(_z1d, _y1d, _x1d, indexing="ij")
shape = _x.shape
EPSILON = 1e-12

# Domain sizes
_lx = 1.0
_ly = 1.0
_lz = 2.0

# shifted grid (origin shifted at the middle of the domain)
_z_shifted = _z - (_lz / 2 + np.min(_z))
_y_shifted = _y - (_ly / 2 + np.min(_y))
_x_shifted = _x - (_lx / 2 + np.min(_x))


@functools.cache
def make_converter(shift_origin=True):
    """Make the converter with or without shifting the origin."""
    return CoordSystem3DConverter(
        _x, _y, _z, _lx, _ly, _lz, shift_origin=shift_origin
    )


@functools.cache
def make_r_h(shift_origin=True):
    """Horizontal (cylindrical) radius centered: sqrt((x - lx/2 - min(x))^2 + (y - ly/2 - min(y))^2), or not centered sqrt(x^2 + y^2)."""
    if shift_origin:
        return np.sqrt(_x_shifted**2 + _y_shifted**2)
    else:
        return np.sqrt(_x**2 + _y**2)


@functools.cache
def make_r_sph_not0(shift_origin=True):
    """Spherical radius centered: sqrt((x - lx/2 - min(x))^2 + (y - ly/2 - min(y))^2 + (z - lz/2 - min(z))^2), or not centered: sqrt(x^2 + y^2 + z^2)."""
    if shift_origin:
        r_sph = np.sqrt(_x_shifted**2 + _y_shifted**2 + _z_shifted**2)
    else:
        r_sph = np.sqrt(_x**2 + _y**2 + _z**2)
    return np.where(r_sph != 0, r_sph, EPSILON)


# ---------------------------------------------------------------------------
# Origin shift test
# ---------------------------------------------------------------------------


def test_origin_shift_coordinates():
    """Test that coordinates are correctly shifted when shift_origin=True."""
    converter = make_converter(shift_origin=False)
    converter_shifted = make_converter()

    assert np.allclose(converter.x, _x)
    assert np.allclose(converter.y, _y)
    assert np.allclose(converter.z, _z)

    assert np.allclose(converter_shifted.x, _x_shifted)
    assert np.allclose(converter_shifted.y, _y_shifted)
    assert np.allclose(converter_shifted.z, _z_shifted)


def test_origin_position():
    """Test that the origin of the shifted grid is at the center of a non-shifted grid that has origin at (0, 0, 0)."""
    n = 5
    x1d = np.linspace(0.0, 1.0, n)
    y1d = np.linspace(0.0, 1.0, n)
    z1d = np.linspace(0.0, 1.0, n)
    z, y, x = np.meshgrid(z1d, y1d, x1d, indexing="ij")
    lx = 1.0
    ly = 1.0
    lz = 1.0

    conv = CoordSystem3DConverter(x, y, z, lx, ly, lz, shift_origin=False)
    conv_shifted = CoordSystem3DConverter(x, y, z, lx, ly, lz, shift_origin=True)

    assert np.allclose(
        np.where(conv_shifted.x == 0.0), np.where(conv.x == lx / 2)
    )
    assert np.allclose(
        np.where(conv_shifted.y == 0.0), np.where(conv.y == ly / 2)
    )
    assert np.allclose(
        np.where(conv_shifted.z == 0.0), np.where(conv.z == lz / 2)
    )


# ---------------------------------------------------------------------------
# compute_r_theta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_compute_r_theta_range(shift_origin):
    """r_theta_shifted must lie in [-pi, pi] and r_theta in [0, pi/2]."""
    converter = make_converter(shift_origin)
    r_theta = converter.compute_r_theta()

    if shift_origin:
        assert np.all(r_theta >= -np.pi)
        assert np.all(r_theta <= np.pi)
    else:
        assert np.all(r_theta >= 0)
        assert np.all(r_theta <= np.pi / 2)


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_compute_r_theta_values(shift_origin, allclose):
    """r_theta should equal arctan2(y, x)."""
    converter = make_converter(shift_origin)
    r_theta = converter.compute_r_theta()

    if shift_origin:
        expected = np.arctan2(_y_shifted, _x_shifted)
    else:
        expected = np.arctan2(_y, _x)
    assert allclose(r_theta, expected)


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_compute_r_theta_origin(shift_origin):
    """r_theta must be 0 when x = y = 0 (on the z-axis)."""
    x = np.zeros((3,))
    y = np.zeros((3,))
    z = np.array([1.0, 0.0, -1.0])
    conv = CoordSystem3DConverter(
        x, y, z, lx=0, ly=0, lz=2, shift_origin=shift_origin
    )
    r_theta = conv.compute_r_theta()
    assert np.all(r_theta == 0.0)


# ---------------------------------------------------------------------------
# compute_cylindrical_components — pure-radial vector
# ---------------------------------------------------------------------------
# A pure-radial (horizontal) vector points in the direction of (x, y, 0),
# i.e. vx = x/r_h, vy = y/r_h, vz = 0.
# In cylindrical coordinates this should give vh = 1, vt = 0, vz = 0.


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
@pytest.mark.parametrize(
    "vector_kind",
    ["pure-radial-h", "pure-azimuthal", "pure-vertical", "pure-spherical-radial"],
)
def test_compute_cylindrical_components(
    shift_origin,
    vector_kind,
    allclose,
):
    converter = make_converter(shift_origin)
    r_h = make_r_h(shift_origin)
    r_sph_not0 = make_r_sph_not0(shift_origin)

    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)

    match vector_kind:
        case "pure-radial-h":
            # Unit vector in the horizontal radial direction
            if shift_origin:
                vx = _x_shifted / r_h_not0
                vy = _y_shifted / r_h_not0
            else:
                vx = _x / r_h_not0
                vy = _y / r_h_not0
            vz = np.zeros(shape)
            vh_exp = np.ones(shape)
            vh_exp[r_h == 0] = 0
            vt_exp = np.zeros(shape)
            vz_exp = np.zeros(shape)

        case "pure-azimuthal":
            # Unit vector in the azimuthal direction: (-y, x, 0) / r_h
            if shift_origin:
                vx = -_y_shifted / r_h_not0
                vy = _x_shifted / r_h_not0
            else:
                vx = -_y / r_h_not0
                vy = _x / r_h_not0
            vz = np.zeros(shape)
            vh_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vt_exp[r_h == 0] = 0
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
            # Unit vector in the spherical radial direction: (x, y, z) / r_sph_not0
            # Cylindrical decomposition: vh = r_h/r_sph_not0, vt = 0, vz = z/r_sph_not0
            if shift_origin:
                vx = _x_shifted / r_sph_not0
                vy = _y_shifted / r_sph_not0
                vz = _z_shifted / r_sph_not0
                vz_exp = _z_shifted / r_sph_not0
            else:
                vx = _x / r_sph_not0
                vy = _y / r_sph_not0
                vz = _z / r_sph_not0
                vz_exp = _z / r_sph_not0
            vh_exp = r_h / r_sph_not0
            vt_exp = np.zeros(shape)

        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    assert allclose(vh, vh_exp), f"vh mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vz_out, vz_exp), f"vz mismatch for {vector_kind}"


# ---------------------------------------------------------------------------
# compute_cylindrical_components — linearity / inverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_cylindrical_preserves_norm(shift_origin, allclose):
    """Cylindrical conversion is a rotation: it must preserve the vector norm."""
    converter = make_converter(shift_origin)
    r_h = make_r_h(shift_origin)

    rng = np.random.default_rng(0)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)
    vx[r_h == 0] = 0
    vy[r_h == 0] = 0

    norm2_cart = vx**2 + vy**2 + vz**2
    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    norm2_cyl = vh**2 + vt**2 + vz_out**2

    assert allclose(norm2_cyl, norm2_cart)


# ---------------------------------------------------------------------------
# compute_radial_component
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_compute_radial_component_pure_radial(shift_origin, allclose):
    """A pure horizontal-radial unit vector should have radial component 1."""
    converter = make_converter(shift_origin)
    r_sph_not0 = make_r_sph_not0(shift_origin)

    if shift_origin:
        vx = _x_shifted / r_sph_not0
        vy = _y_shifted / r_sph_not0
        vz = _z_shifted / r_sph_not0
    else:
        vx = _x / r_sph_not0
        vy = _y / r_sph_not0
        vz = _z / r_sph_not0
    vr = converter.compute_radial_component(vx, vy, vz)
    assert allclose(vr, np.ones(shape))


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_compute_radial_component_pure_azimuthal(shift_origin, allclose):
    """A pure azimuthal unit vector is perpendicular to r_h → radial component 0."""

    converter = make_converter(shift_origin)
    r_h = make_r_h(shift_origin)

    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)

    if shift_origin:
        vx = -_y_shifted / r_h_not0
        vy = _x_shifted / r_h_not0
    else:
        vx = -_y / r_h_not0
        vy = _x / r_h_not0
    vz = np.zeros(shape)
    vr = converter.compute_radial_component(vx, vy, vz)

    assert allclose(vr, np.zeros(shape))


# ---------------------------------------------------------------------------
# compute_spherical_components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
@pytest.mark.parametrize(
    "vector_kind",
    ["pure-spherical-radial", "pure-azimuthal", "pure-polar"],
)
def test_compute_spherical_components(
    shift_origin,
    vector_kind,
    allclose,
):
    converter = make_converter(shift_origin)
    r_h = make_r_h(shift_origin)
    r_sph_not0 = make_r_sph_not0(shift_origin)

    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)

    match vector_kind:
        case "pure-spherical-radial":
            if shift_origin:
                # Unit vector along spherical r with shifted origin
                vx = _x_shifted / r_sph_not0
                vy = _y_shifted / r_sph_not0
                vz = _z_shifted / r_sph_not0
            else:
                # Unit vector along spherical r: (x, y, z)/r_sph_not0
                vx = _x / r_sph_not0
                vy = _y / r_sph_not0
                vz = _z / r_sph_not0
            vr_exp = np.ones(shape)
            vt_exp = np.zeros(shape)  # azimuthal
            vp_exp = np.zeros(shape)  # polar

        case "pure-azimuthal":
            if shift_origin:
                # Unit vector along azimuthal phi with shifted origin
                vx = -_y_shifted / r_h_not0
                vy = _x_shifted / r_h_not0
            else:
                # Unit vector along azimuthal phi: (-y, x, 0)/r_h
                vx = -_y / r_h_not0
                vy = _x / r_h_not0
            vz = np.zeros(shape)
            vr_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vt_exp[r_h == 0] = 0
            vp_exp = np.zeros(shape)

        case "pure-polar":
            if shift_origin:
                # Unit vector along polar theta (e_theta) with shifted origin
                vx = _x_shifted * _z_shifted / (r_sph_not0 * r_h_not0)
                vy = _y_shifted * _z_shifted / (r_sph_not0 * r_h_not0)
            else:
                # Unit vector along polar theta (e_theta): (x*z, y*z, -r_h^2) / (r_sph_not0 * r_h)
                vx = _x * _z / (r_sph_not0 * r_h_not0)
                vy = _y * _z / (r_sph_not0 * r_h_not0)
            vz = -(r_h**2) / (r_sph_not0 * r_h_not0)
            vr_exp = np.zeros(shape)
            vt_exp = np.zeros(shape)
            vp_exp = np.ones(shape)
            vp_exp[r_h == 0] = 0

        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    assert allclose(vr, vr_exp), f"vr mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vp, vp_exp), f"vp mismatch for {vector_kind}"


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_spherical_preserves_norm(shift_origin, allclose):
    """Spherical conversion is a rotation: it must preserve the vector norm."""
    converter = make_converter(shift_origin)
    r_h = make_r_h(shift_origin)

    rng = np.random.default_rng(1)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    vx[r_h == 0] = 0
    vy[r_h == 0] = 0

    norm2_cart = vx**2 + vy**2 + vz**2
    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    norm2_sph = vr**2 + vt**2 + vp**2

    assert allclose(norm2_sph, norm2_cart)


@pytest.mark.parametrize(
    "shift_origin",
    [False, True],
)
def test_spherical_radial_equals_radial_component(shift_origin, allclose):
    """The spherical vr component must equal compute_radial_component."""
    converter = make_converter(shift_origin)

    rng = np.random.default_rng(2)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    vr_sph, _, _ = converter.compute_spherical_components(vx, vy, vz)
    vr_direct = converter.compute_radial_component(vx, vy, vz)

    assert allclose(vr_sph, vr_direct)
