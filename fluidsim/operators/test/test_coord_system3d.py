import pytest
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


@pytest.fixture(scope="module")
def converter():
    """Converter with shift_origin=False (origin at corner)."""
    return CoordSystem3DConverter(_x, _y, _z, _lx, _ly, _lz, shift_origin=False)


@pytest.fixture(scope="module")
def converter_shifted():
    return CoordSystem3DConverter(_x, _y, _z, _lx, _ly, _lz, shift_origin=True)


@pytest.fixture(scope="module")
def r_h():
    """Horizontal (cylindrical) radius sqrt(x^2 + y^2)."""
    return np.sqrt(_x**2 + _y**2)


@pytest.fixture(scope="module")
def r_h_shifted():
    """Horizontal (cylindrical) radius sqrt((x - lx/2 - min(x))^2 + (y - ly/2 - min(y))^2), i.e. centered."""
    return np.sqrt(_x_shifted**2 + _y_shifted**2)


@pytest.fixture(scope="module")
def r_sph_not0():
    """Spherical radius sqrt(x^2 + y^2 + z^2)."""
    r_sph = np.sqrt(_x**2 + _y**2 + _z**2)
    return np.where(r_sph != 0, r_sph, EPSILON)


@pytest.fixture(scope="module")
def r_sph_not0_shifted():
    """Spherical radius sqrt((x - lx/2 - min(x))^2 + (y - ly/2 - min(y))^2 + (z - lz/2 - min(z))^2)."""
    r_sph = np.sqrt(_x_shifted**2 + _y_shifted**2 + _z_shifted**2)
    return np.where(r_sph != 0, r_sph, EPSILON)


# ---------------------------------------------------------------------------
# Origin shift test
# ---------------------------------------------------------------------------


def test_origin_shift_coordinates(converter, converter_shifted):
    """Test that coordinates are correctly shifted when shift_origin=True."""

    assert np.allclose(converter.x, _x)
    assert np.allclose(converter.y, _y)
    assert np.allclose(converter.z, _z)

    assert np.allclose(converter_shifted.x, _x_shifted)
    assert np.allclose(converter_shifted.y, _y_shifted)
    assert np.allclose(converter_shifted.z, _z_shifted)


# ---------------------------------------------------------------------------
# compute_r_theta
# ---------------------------------------------------------------------------


def test_compute_r_theta_range(converter, converter_shifted):
    """r_theta_shifted must lie in [-pi, pi] and r_theta in [0, pi/2]."""
    r_theta = converter.compute_r_theta()
    r_theta_shifted = converter_shifted.compute_r_theta()
    assert np.all(r_theta >= 0)
    assert np.all(r_theta <= np.pi / 2)
    assert np.all(r_theta_shifted >= -np.pi)
    assert np.all(r_theta_shifted <= np.pi)


def test_compute_r_theta_values(converter, converter_shifted, allclose):
    """r_theta should equal arctan2(y, x)."""
    r_theta = converter.compute_r_theta()
    expected = np.arctan2(_y, _x)
    r_theta_shifted = converter_shifted.compute_r_theta()
    expected_shifted = np.arctan2(_y_shifted, _x_shifted)
    assert allclose(r_theta, expected)
    assert allclose(r_theta_shifted, expected_shifted)


def test_compute_r_theta_origin():
    """r_theta must be 0 when x = y = 0 (on the z-axis)."""
    x = np.zeros((3,))
    y = np.zeros((3,))
    z = np.array([1.0, 0.0, -1.0])
    conv = CoordSystem3DConverter(x, y, z, lx=0, ly=0, lz=2, shift_origin=False)
    r_theta = conv.compute_r_theta()
    conv_shifted = CoordSystem3DConverter(
        x, y, z, lx=0, ly=0, lz=2, shift_origin=True
    )
    r_theta_shifted = conv_shifted.compute_r_theta()
    assert np.all(r_theta == 0.0)
    assert np.all(r_theta_shifted == 0.0)


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
    vector_kind,
    converter,
    converter_shifted,
    r_h,
    r_h_shifted,
    r_sph_not0,
    r_sph_not0_shifted,
    allclose,
):
    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)
    r_h_not0_shifted = np.where(r_h_shifted != 0, r_h_shifted, EPSILON)

    match vector_kind:
        case "pure-radial-h":
            # Unit vector in the horizontal radial direction
            vx = _x / r_h_not0
            vy = _y / r_h_not0
            vz = np.zeros(shape)
            vh_exp = np.ones(shape)
            vh_exp[r_h == 0] = 0
            vt_exp = np.zeros(shape)
            vz_exp = np.zeros(shape)
            vx_shifted = _x_shifted / r_h_not0_shifted
            vy_shifted = _y_shifted / r_h_not0_shifted
            vz_shifted = np.zeros(shape)
            vh_exp_shifted = np.ones(shape)
            vh_exp_shifted[r_h_shifted == 0] = 0
            vt_exp_shifted = np.zeros(shape)
            vz_exp_shifted = np.zeros(shape)

        case "pure-azimuthal":
            # Unit vector in the azimuthal direction: (-y, x, 0) / r_h
            vx = -_y / r_h_not0
            vy = _x / r_h_not0
            vz = np.zeros(shape)
            vh_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vt_exp[r_h == 0] = 0
            vz_exp = np.zeros(shape)
            vx_shifted = -_y_shifted / r_h_not0_shifted
            vy_shifted = _x_shifted / r_h_not0_shifted
            vz_shifted = np.zeros(shape)
            vh_exp_shifted = np.zeros(shape)
            vt_exp_shifted = np.ones(shape)
            vt_exp_shifted[r_h_shifted == 0] = 0
            vz_exp_shifted = np.zeros(shape)

        case "pure-vertical":
            # Unit vector along z
            vx = np.zeros(shape)
            vy = np.zeros(shape)
            vz = np.ones(shape)
            vh_exp = np.zeros(shape)
            vt_exp = np.zeros(shape)
            vz_exp = np.ones(shape)
            vx_shifted = np.zeros(shape)
            vy_shifted = np.zeros(shape)
            vz_shifted = np.ones(shape)
            vh_exp_shifted = np.zeros(shape)
            vt_exp_shifted = np.zeros(shape)
            vz_exp_shifted = np.ones(shape)

        case "pure-spherical-radial":
            # Unit vector in the spherical radial direction: (x, y, z) / r_sph_not0
            # Cylindrical decomposition: vh = r_h/r_sph_not0, vt = 0, vz = z/r_sph_not0
            vx = _x / r_sph_not0
            vy = _y / r_sph_not0
            vz = _z / r_sph_not0
            vh_exp = r_h / r_sph_not0
            vt_exp = np.zeros(shape)
            vz_exp = _z / r_sph_not0
            vx_shifted = _x_shifted / r_sph_not0_shifted
            vy_shifted = _y_shifted / r_sph_not0_shifted
            vz_shifted = _z_shifted / r_sph_not0_shifted
            vh_exp_shifted = r_h_shifted / r_sph_not0_shifted
            vt_exp_shifted = np.zeros(shape)
            vz_exp_shifted = _z_shifted / r_sph_not0_shifted
        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    vh_shifted, vt_shifted, vz_out_shifted = (
        converter_shifted.compute_cylindrical_components(
            vx_shifted, vy_shifted, vz_shifted
        )
    )
    assert allclose(vh, vh_exp), f"vh mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vz_out, vz_exp), f"vz mismatch for {vector_kind}"
    assert allclose(vh_shifted, vh_exp_shifted), f"vh mismatch for {vector_kind}"
    assert allclose(vt_shifted, vt_exp_shifted), f"vt mismatch for {vector_kind}"
    assert allclose(vz_out_shifted, vz_exp_shifted), (
        f"vz mismatch for {vector_kind}"
    )


# ---------------------------------------------------------------------------
# compute_cylindrical_components — linearity / inverse
# ---------------------------------------------------------------------------


def test_cylindrical_preserves_norm(
    converter, converter_shifted, r_h, r_h_shifted, allclose
):
    """Cylindrical conversion is a rotation: it must preserve the vector norm."""
    rng = np.random.default_rng(0)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)
    vx[r_h == 0] = 0
    vy[r_h == 0] = 0
    vx_shifted = rng.standard_normal(shape)
    vy_shifted = rng.standard_normal(shape)
    vz_shifted = rng.standard_normal(shape)
    vx_shifted[r_h_shifted == 0] = 0
    vy_shifted[r_h_shifted == 0] = 0

    norm2_cart = vx**2 + vy**2 + vz**2
    vh, vt, vz_out = converter.compute_cylindrical_components(vx, vy, vz)
    norm2_cyl = vh**2 + vt**2 + vz_out**2

    norm2_cart_shifted = vx_shifted**2 + vy_shifted**2 + vz_shifted**2
    vh_shifted, vt_shifted, vz_out_shifted = (
        converter_shifted.compute_cylindrical_components(
            vx_shifted, vy_shifted, vz_shifted
        )
    )
    norm2_cyl_shifted = vh_shifted**2 + vt_shifted**2 + vz_out_shifted**2

    assert allclose(norm2_cyl, norm2_cart)
    assert allclose(norm2_cyl_shifted, norm2_cart_shifted)


# ---------------------------------------------------------------------------
# compute_radial_component
# ---------------------------------------------------------------------------


def test_compute_radial_component_pure_radial(
    converter, converter_shifted, r_sph_not0, r_sph_not0_shifted, allclose
):
    """A pure horizontal-radial unit vector should have radial component 1."""
    vx = _x / r_sph_not0
    vy = _y / r_sph_not0
    vz = _z / r_sph_not0
    vr = converter.compute_radial_component(vx, vy, vz)
    vx_shifted = _x_shifted / r_sph_not0_shifted
    vy_shifted = _y_shifted / r_sph_not0_shifted
    vz_shifted = _z_shifted / r_sph_not0_shifted
    vr_shifted = converter_shifted.compute_radial_component(
        vx_shifted, vy_shifted, vz_shifted
    )
    assert allclose(vr, np.ones(shape))
    assert allclose(vr_shifted, np.ones(shape))


def test_compute_radial_component_pure_azimuthal(
    converter, converter_shifted, r_h, r_h_shifted, allclose
):
    """A pure azimuthal unit vector is perpendicular to r_h → radial component 0."""

    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)

    r_h_not0_shifted = np.where(r_h_shifted != 0, r_h_shifted, EPSILON)

    vx = -_y / r_h_not0
    vy = _x / r_h_not0
    vz = np.zeros(shape)
    vr = converter.compute_radial_component(vx, vy, vz)

    vx_shifted = -_y_shifted / r_h_not0_shifted
    vy_shifted = _x_shifted / r_h_not0_shifted
    vz_shifted = np.zeros(shape)
    vr_shifted = converter_shifted.compute_radial_component(
        vx_shifted, vy_shifted, vz_shifted
    )
    assert allclose(vr, np.zeros(shape))
    assert allclose(vr_shifted, np.zeros(shape))


# ---------------------------------------------------------------------------
# compute_spherical_components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector_kind",
    ["pure-spherical-radial", "pure-azimuthal", "pure-polar"],
)
def test_compute_spherical_components(
    vector_kind,
    converter,
    converter_shifted,
    r_h,
    r_h_shifted,
    r_sph_not0,
    r_sph_not0_shifted,
    allclose,
):
    r_h_not0 = np.where(r_h != 0, r_h, EPSILON)

    r_h_not0_shifted = np.where(r_h_shifted != 0, r_h_shifted, EPSILON)

    match vector_kind:
        case "pure-spherical-radial":
            # Unit vector along spherical r: (x, y, z)/r_sph_not0
            vx = _x / r_sph_not0
            vy = _y / r_sph_not0
            vz = _z / r_sph_not0
            vr_exp = np.ones(shape)
            vt_exp = np.zeros(shape)  # azimuthal
            vp_exp = np.zeros(shape)  # polar
            # Unit vector along spherical r with shifted origin
            vx_shifted = _x_shifted / r_sph_not0_shifted
            vy_shifted = _y_shifted / r_sph_not0_shifted
            vz_shifted = _z_shifted / r_sph_not0_shifted
            vr_exp_shifted = np.ones(shape)
            vt_exp_shifted = np.zeros(shape)  # azimuthal
            vp_exp_shifted = np.zeros(shape)  # polar

        case "pure-azimuthal":
            # Unit vector along azimuthal phi: (-y, x, 0)/r_h
            vx = -_y / r_h_not0
            vy = _x / r_h_not0
            vz = np.zeros(shape)
            vr_exp = np.zeros(shape)
            vt_exp = np.ones(shape)
            vt_exp[r_h == 0] = 0
            vp_exp = np.zeros(shape)
            # Unit vector along azimuthal phi with shifted origin
            vx_shifted = -_y_shifted / r_h_not0_shifted
            vy_shifted = _x_shifted / r_h_not0_shifted
            vz_shifted = np.zeros(shape)
            vr_exp_shifted = np.zeros(shape)
            vt_exp_shifted = np.ones(shape)
            vt_exp_shifted[r_h_shifted == 0] = 0
            vp_exp_shifted = np.zeros(shape)

        case "pure-polar":
            # Unit vector along polar theta (e_theta): (x*z, y*z, -r_h^2) / (r_sph_not0 * r_h)
            vx = _x * _z / (r_sph_not0 * r_h_not0)
            vy = _y * _z / (r_sph_not0 * r_h_not0)
            vz = -(r_h**2) / (r_sph_not0 * r_h_not0)
            vr_exp = np.zeros(shape)
            vt_exp = np.zeros(shape)
            vp_exp = np.ones(shape)
            vp_exp[r_h == 0] = 0
            # Unit vector along polar theta (e_theta) with shifted origin
            vx_shifted = (
                _x_shifted * _z_shifted / (r_sph_not0_shifted * r_h_not0_shifted)
            )
            vy_shifted = (
                _y_shifted * _z_shifted / (r_sph_not0_shifted * r_h_not0_shifted)
            )
            vz_shifted = -(r_h_shifted**2) / (
                r_sph_not0_shifted * r_h_not0_shifted
            )
            vr_exp_shifted = np.zeros(shape)
            vt_exp_shifted = np.zeros(shape)
            vp_exp_shifted = np.ones(shape)
            vp_exp_shifted[r_h_shifted == 0] = 0

        case _:
            raise ValueError(f"Unknown vector_kind: {vector_kind}")

    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    vr_shifted, vt_shifted, vp_shifted = (
        converter_shifted.compute_spherical_components(
            vx_shifted, vy_shifted, vz_shifted
        )
    )
    assert allclose(vr, vr_exp), f"vr mismatch for {vector_kind}"
    assert allclose(vt, vt_exp), f"vt mismatch for {vector_kind}"
    assert allclose(vp, vp_exp), f"vp mismatch for {vector_kind}"
    assert allclose(vr_shifted, vr_exp_shifted), f"vr mismatch for {vector_kind}"
    assert allclose(vt_shifted, vt_exp_shifted), f"vt mismatch for {vector_kind}"
    assert allclose(vp_shifted, vp_exp_shifted), f"vp mismatch for {vector_kind}"


def test_spherical_preserves_norm(
    converter, converter_shifted, r_h, r_h_shifted, allclose
):
    """Spherical conversion is a rotation: it must preserve the vector norm."""
    rng = np.random.default_rng(1)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)
    vx_shifted = rng.standard_normal(shape)
    vy_shifted = rng.standard_normal(shape)
    vz_shifted = rng.standard_normal(shape)

    vx[r_h == 0] = 0
    vy[r_h == 0] = 0

    vx_shifted[r_h_shifted == 0] = 0
    vy_shifted[r_h_shifted == 0] = 0

    norm2_cart = vx**2 + vy**2 + vz**2
    vr, vt, vp = converter.compute_spherical_components(vx, vy, vz)
    norm2_sph = vr**2 + vt**2 + vp**2

    norm2_cart_shifted = vx_shifted**2 + vy_shifted**2 + vz_shifted**2
    vr_shifted, vt_shifted, vp_shifted = (
        converter_shifted.compute_spherical_components(
            vx_shifted, vy_shifted, vz_shifted
        )
    )
    norm2_sph_shifted = vr_shifted**2 + vt_shifted**2 + vp_shifted**2

    assert allclose(norm2_sph, norm2_cart)
    assert allclose(norm2_sph_shifted, norm2_cart_shifted)


def test_spherical_radial_equals_radial_component(
    converter, converter_shifted, allclose
):
    """The spherical vr component must equal compute_radial_component."""
    rng = np.random.default_rng(2)
    vx = rng.standard_normal(shape)
    vy = rng.standard_normal(shape)
    vz = rng.standard_normal(shape)

    vr_sph, _, _ = converter.compute_spherical_components(vx, vy, vz)
    vr_direct = converter.compute_radial_component(vx, vy, vz)
    vr_sph_shifted, _, _ = converter_shifted.compute_spherical_components(
        vx, vy, vz
    )
    vr_direct_shifted = converter_shifted.compute_radial_component(vx, vy, vz)

    assert allclose(vr_sph, vr_direct)
    assert allclose(vr_sph_shifted, vr_direct_shifted)
