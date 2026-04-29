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
    return MockOperator(nx=100, ny=100, nz=100, Lx=1.0, Ly=1.0, Lz=1.0)


@pytest.fixture(scope="module")
def spatial_avg(mock_oper):
    """Create SpatialAverage instance."""
    return SpatialAverage(mock_oper, dr=2.0, drh=2.0, dz=1.0, shift_origin=True)


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
    but with discretization error. We test correlation instead of exact match.
    """
    field = spatial_avg.r.copy()
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert np.all(np.diff(field_avg) > 0), "Average should increase with radius"

    assert np.allclose(field_avg[4:], r_centers[4:], rtol=0.03)


def test_radial_average_quadratic_field(spatial_avg):
    """Test radial average of f(r) = r^2.

    Similar to linear case, we test correlation and approximate scaling.
    """
    field = spatial_avg.r**2
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    expected = r_centers**2

    assert np.allclose(field_avg[4:], expected[4:], rtol=0.06)
