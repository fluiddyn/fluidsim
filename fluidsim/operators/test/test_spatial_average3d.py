import pytest
import numpy as np

from fluidsim.operators.spatial_average3d import SpatialAverage


class MockOperator:
    """Minimal operator for testing spatial averaging."""

    def __init__(self, nx, ny, nz, Lx, Ly, Lz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.delta = Lx / nx  # Assume uniform spacing

        # Create a simple uniform grid
        x1d = np.linspace(0, Lx, nx, endpoint=False)
        y1d = np.linspace(0, Ly, ny, endpoint=False)
        z1d = np.linspace(0, Lz, nz, endpoint=False)

        self.Z, self.Y, self.X = np.meshgrid(z1d, y1d, x1d, indexing="ij")
        self.shapeX_loc = self.X.shape

    def get_XYZ_loc(self):
        """Return local coordinates (no MPI in tests)."""
        return self.X, self.Y, self.Z


@pytest.fixture(scope="module")
def mock_oper():
    """Create a small 3D operator for testing."""
    return MockOperator(nx=20, ny=20, nz=20, Lx=1.0, Ly=1.0, Lz=1.0)


@pytest.fixture(scope="module")
def spatial_avg(mock_oper):
    """Create SpatialAverage instance."""
    return SpatialAverage(mock_oper, dr=1.0, drh=1.0, dz=1.0, shift_origin=True)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_coordinate_shapes(spatial_avg, mock_oper):
    """Test that computed coordinates have correct shapes."""
    shape = mock_oper.X.shape
    assert spatial_avg.r.shape == shape
    assert spatial_avg.rho.shape == shape


def test_bins_positive(spatial_avg):
    """Test that bin centers are positive and sorted."""
    assert np.all(spatial_avg.r_centers > 0)
    assert np.all(np.diff(spatial_avg.r_centers) > 0)  # monotonically increasing
    assert np.all(spatial_avg.rho_centers >= 0)
    assert np.all(np.diff(spatial_avg.rho_centers) > 0)


def test_steps(spatial_avg):
    """Test that computed coordinates have correct shapes."""
    assert np.allclose(
        spatial_avg.r_centers[1] - spatial_avg.r_centers[0],
        spatial_avg.deltar,
        rtol=1e-12,
    )
    assert np.allclose(
        spatial_avg.rho_centers[1] - spatial_avg.rho_centers[0],
        spatial_avg.deltarh,
        rtol=1e-12,
    )
    assert np.allclose(
        spatial_avg.z_centers[1] - spatial_avg.z_centers[0],
        spatial_avg.deltaz,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Radial average tests
# ---------------------------------------------------------------------------


def test_radial_average_constant_field(spatial_avg, allclose):
    """Radial average of a constant field should be constant."""
    field = np.ones_like(spatial_avg.X) * 5.0
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert allclose(field_avg, 5.0, rtol=1e-10)


def test_radial_average_radial_field(spatial_avg):
    """Test radial average of f(r) = r.

    The average of r over a spherical shell is approximately r_center,
    but with discretization error.
    """
    field = spatial_avg.r.copy()
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert np.all(np.diff(field_avg) > 0), "Average should increase with radius"

    assert np.allclose(field_avg[0], r_centers[0], rtol=1e-12)
    assert np.allclose(field_avg[4:], r_centers[4:], rtol=0.03)
    assert np.allclose(field_avg[6:-3], r_centers[6:-3], rtol=0.009)


def test_radial_average_vector_field(spatial_avg):
    """Test radial average of a 3D vector field."""
    shape = spatial_avg.X.shape
    v_r = np.ones(shape)
    v_theta = np.ones(shape) * 2
    v_phi = np.ones(shape) * 3
    vector_field = np.array([v_r, v_theta, v_phi])

    r_centers, v_avg = spatial_avg.compute_radial_average(vector_field)

    assert v_avg.shape == (3, spatial_avg.nr)
    assert np.allclose(v_avg[0], 1.0, rtol=1e-10)
    assert np.allclose(v_avg[1], 2.0, rtol=1e-10)
    assert np.allclose(v_avg[2], 3.0, rtol=1e-10)


def test_radial_average_with_std(spatial_avg):
    """Test that standard deviation is returned when requested."""
    field = np.random.randn(*spatial_avg.X.shape)
    r_centers, field_avg, field_std = spatial_avg.compute_radial_average(
        field, return_std=True
    )

    assert field_avg.shape == (spatial_avg.nr,)
    assert field_std.shape == (spatial_avg.nr,)
    assert np.all(field_std >= 0)


# ---------------------------------------------------------------------------
# Azimuthal average tests
# ---------------------------------------------------------------------------


def test_azimuthal_average_constant_field(spatial_avg, allclose):
    """Azimuthal average of a constant field should be constant."""
    field = np.ones_like(spatial_avg.X) * 7.0
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    assert allclose(field_avg, 7.0, rtol=1e-10)


@pytest.mark.parametrize("azimut_var", ["z", "rho"])
def test_azimuthal_average_var_dependent(spatial_avg, azimut_var):
    """Test azimuthal average of f(z) = z and f(rho) = rho.

    We compute the expected average by explicitly averaging the actual z values
    in each bin, accounting for the discrete grid.
    """
    if azimut_var == "z":
        field = spatial_avg.Z.copy()

        rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
            field
        )

        expected_avg = np.zeros((spatial_avg._nz, spatial_avg.nrh))
        for i, z_val in enumerate(z_centers):
            expected_avg[i, :] += z_val

        assert np.allclose(
            field_avg,
            expected_avg,
            rtol=1e-12,
        )

    elif azimut_var == "rho":
        field = spatial_avg.rho.copy()

        rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
            field
        )

        expected_avg = np.zeros((spatial_avg._nz, spatial_avg.nrh))
        for i, rho_val in enumerate(rho_centers):
            expected_avg[:, i] += rho_val

        assert np.allclose(
            field_avg[:, 0],
            expected_avg[:, 0],
            rtol=1e-12,
        )
        assert np.allclose(
            field_avg[:, 4:],
            expected_avg[:, 4:],
            rtol=0.03,
        )


def test_azimuthal_average_vector_field(spatial_avg):
    """Test azimuthal average of a 3D vector field."""
    shape = spatial_avg.X.shape
    v_rho = np.ones(shape) * 1.5
    v_theta = np.ones(shape) * 2.5
    v_z = np.ones(shape) * 3.5
    vector_field = np.array([v_rho, v_theta, v_z])

    rho_centers, z_centers, v_avg = spatial_avg.compute_azimuthal_average(
        vector_field
    )

    assert v_avg.shape == (3, spatial_avg._nz, spatial_avg.nrh)
    assert np.allclose(v_avg[0], 1.5, rtol=1e-10)
    assert np.allclose(v_avg[1], 2.5, rtol=1e-10)
    assert np.allclose(v_avg[2], 3.5, rtol=1e-10)


def test_azimuthal_average_with_std(spatial_avg):
    """Test that standard deviation is returned when requested."""
    field = np.random.randn(*spatial_avg.X.shape)
    rho_centers, z_centers, field_avg, field_std = (
        spatial_avg.compute_azimuthal_average(field, return_std=True)
    )

    assert field_avg.shape == (spatial_avg._nz, spatial_avg.nrh)
    assert field_std.shape == (spatial_avg._nz, spatial_avg.nrh)
    assert np.all(field_std >= 0)


# ---------------------------------------------------------------------------
# Volume weights tests
# ---------------------------------------------------------------------------


def test_compute_volume_weights_sum(spatial_avg, mock_oper):
    """Test that sum of volume weights equals total domain volume."""
    weights = spatial_avg.compute_volume_weights()
    total_volume = np.sum(weights)
    expected_volume = mock_oper.Lx * mock_oper.Ly * mock_oper.Lz
    assert np.allclose(total_volume, expected_volume, rtol=1e-10)


def test_compute_volume_weights_field_radial_average_sum(spatial_avg, mock_oper):
    """Test that sum of volume weights field radial average equals total domain volume."""
    weights = spatial_avg.compute_volume_weights()
    _, weights_avg = spatial_avg.compute_radial_average(weights)
    volumes_averages = np.mean(weights_avg)
    expected_volumes_average = mock_oper.delta**3
    assert np.allclose(volumes_averages, expected_volumes_average, rtol=0.1)


def test_compute_volume_weights_field_azimuthal_average_sum(
    spatial_avg, mock_oper
):
    """Test that sum of volume weights field radial average equals total domain volume."""
    weights = spatial_avg.compute_volume_weights()
    _, _, weights_avg = spatial_avg.compute_azimuthal_average(weights)
    volumes_averages = np.mean(weights_avg)
    expected_volumes_average = mock_oper.delta**3
    assert np.allclose(volumes_averages, expected_volumes_average, rtol=0.1)
