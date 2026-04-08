"""Tests for spatial_average3d module."""

import pytest
import numpy as np

from fluidsim.operators.spatial_average3d import SpatialAverage


# Mock operator for testing (without full fluidsim dependency)
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
        z1d = np.linspace(-Lz/2, Lz/2, nz, endpoint=False)
        
        self.Z, self.Y, self.X = np.meshgrid(z1d, y1d, x1d, indexing='ij')
        self.shapeX_loc = self.X.shape
    
    def get_XYZ_loc(self):
        """Return local coordinates (no MPI in tests)."""
        return self.X, self.Y, self.Z


@pytest.fixture(scope="module")
def mock_oper():
    """Create a small 3D operator for testing."""
    return MockOperator(nx=8, ny=8, nz=8, Lx=2.0, Ly=2.0, Lz=2.0)


@pytest.fixture(scope="module")
def spatial_avg(mock_oper):
    """Create SpatialAverage instance."""
    return SpatialAverage(mock_oper, nr=10, nrh=8, nz=8)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_initialization(spatial_avg, mock_oper):
    """Test that spatial average initializes correctly."""
    assert spatial_avg.nr == 10
    assert spatial_avg.nrh == 8
    assert spatial_avg.nz == 8
    assert spatial_avg.X.shape == mock_oper.X.shape
    assert hasattr(spatial_avg, 'r')
    assert hasattr(spatial_avg, 'rho')
    assert hasattr(spatial_avg, 'phi')


def test_coordinate_shapes(spatial_avg, mock_oper):
    """Test that computed coordinates have correct shapes."""
    shape = mock_oper.X.shape
    assert spatial_avg.r.shape == shape
    assert spatial_avg.rho.shape == shape
    assert spatial_avg.phi.shape == shape


def test_bins_positive(spatial_avg):
    """Test that bin centers are positive and sorted."""
    assert np.all(spatial_avg.r_centers > 0)
    assert np.all(np.diff(spatial_avg.r_centers) > 0)  # monotonically increasing
    assert np.all(spatial_avg.rho_centers >= 0)
    assert np.all(np.diff(spatial_avg.rho_centers) > 0)


# ---------------------------------------------------------------------------
# Radial average tests
# ---------------------------------------------------------------------------


def test_radial_average_constant_field(spatial_avg, allclose):
    """Radial average of a constant field should be constant."""
    field = np.ones_like(spatial_avg.X) * 5.0
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    # All bins should have approximately the same value
    assert allclose(field_avg, 5.0, rtol=1e-10)


def test_radial_average_radial_field(spatial_avg):
    """Test radial average of f(r) = r.
    
    The average of r over a spherical shell is approximately r_center,
    but with discretization error. We test correlation instead of exact match.
    """
    field = spatial_avg.r.copy()
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    # Test that field_avg increases monotonically with r_centers
    assert np.all(np.diff(field_avg) > 0), "Average should increase with radius"
    
    # Test correlation: field_avg should be highly correlated with r_centers
    correlation = np.corrcoef(field_avg, r_centers)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"
    
    # Test that the ratio is close to 1 (within discretization error)
    # Skip first bin where r is very small
    ratio = field_avg[1:] / r_centers[1:]
    assert np.allclose(ratio, 1.0, rtol=0.2), "Ratio should be close to 1"


def test_radial_average_quadratic_field(spatial_avg):
    """Test radial average of f(r) = r^2.
    
    Similar to linear case, we test correlation and approximate scaling.
    """
    field = spatial_avg.r**2
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    expected = r_centers**2
    
    # Test correlation
    correlation = np.corrcoef(field_avg, expected)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"
    
    # Test approximate scaling (skip first bin)
    ratio = field_avg[1:] / expected[1:]
    assert np.allclose(ratio, 1.0, rtol=0.25), "Ratio should be close to 1"


def test_radial_average_vector_field(spatial_avg):
    """Test radial average of a 3D vector field."""
    shape = spatial_avg.X.shape
    vx = np.ones(shape)
    vy = np.ones(shape) * 2
    vz = np.ones(shape) * 3
    vector_field = np.array([vx, vy, vz])
    
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
    assert np.all(field_std >= 0)  # Std must be non-negative


def test_radial_average_zero_field(spatial_avg, allclose):
    """Radial average of zero field should be zero."""
    field = np.zeros_like(spatial_avg.X)
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    assert allclose(field_avg, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Azimuthal average tests
# ---------------------------------------------------------------------------


def test_azimuthal_average_constant_field(spatial_avg, allclose):
    """Azimuthal average of a constant field should be constant."""
    field = np.ones_like(spatial_avg.X) * 7.0
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    assert allclose(field_avg, 7.0, rtol=1e-10)


def test_azimuthal_average_z_dependent(spatial_avg):
    """Test azimuthal average of f(z) = z.
    
    For each z bin, the average should be close to z_center,
    but with discretization error.
    """
    field = spatial_avg.Z.copy()
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    # Average over rho (each column corresponds to a z level)
    avg_over_rho = np.mean(field_avg, axis=0)
    
    # Test correlation
    correlation = np.corrcoef(avg_over_rho, z_centers)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"
    
    # Test that values are close (with generous tolerance for discretization)
    # The issue is that bins near the boundaries have fewer points
    # so we test only the middle bins
    n_skip = 2  # Skip edge bins
    assert np.allclose(
        avg_over_rho[n_skip:-n_skip], 
        z_centers[n_skip:-n_skip], 
        rtol=0.3
    ), "Average should be close to z_centers in middle bins"


def test_azimuthal_average_rho_dependent(spatial_avg, allclose):
    """Test azimuthal average of f(rho) = rho."""
    field = spatial_avg.rho.copy()
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    # For each rho bin, average should equal rho_center
    # Average over z (each row should be same)
    avg_over_z = np.mean(field_avg, axis=1)
    
    # Test correlation
    correlation = np.corrcoef(avg_over_z, rho_centers)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"
    
    # Skip first bin (near axis) where discretization is worst
    assert np.allclose(avg_over_z[1:], rho_centers[1:], rtol=0.2)


def test_azimuthal_average_vector_field(spatial_avg):
    """Test azimuthal average of a 3D vector field."""
    shape = spatial_avg.X.shape
    vx = np.ones(shape) * 1.5
    vy = np.ones(shape) * 2.5
    vz = np.ones(shape) * 3.5
    vector_field = np.array([vx, vy, vz])
    
    rho_centers, z_centers, v_avg = spatial_avg.compute_azimuthal_average(vector_field)
    
    assert v_avg.shape == (3, spatial_avg.nrh, spatial_avg.nz)
    assert np.allclose(v_avg[0], 1.5, rtol=1e-10)
    assert np.allclose(v_avg[1], 2.5, rtol=1e-10)
    assert np.allclose(v_avg[2], 3.5, rtol=1e-10)


def test_azimuthal_average_with_std(spatial_avg):
    """Test that standard deviation is returned when requested."""
    field = np.random.randn(*spatial_avg.X.shape)
    rho_centers, z_centers, field_avg, field_std = spatial_avg.compute_azimuthal_average(
        field, return_std=True
    )
    
    assert field_avg.shape == (spatial_avg.nrh, spatial_avg.nz)
    assert field_std.shape == (spatial_avg.nrh, spatial_avg.nz)
    assert np.all(field_std >= 0)


def test_azimuthal_average_zero_field(spatial_avg, allclose):
    """Azimuthal average of zero field should be zero."""
    field = np.zeros_like(spatial_avg.X)
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    assert allclose(field_avg, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Physics tests: sin(phi) weighting
# ---------------------------------------------------------------------------


def test_radial_average_sin_phi_weighting(spatial_avg):
    """Test that sin(phi) weighting is correctly applied."""
    # Field = 1 everywhere
    field = np.ones_like(spatial_avg.X)
    
    # Compute radial average (uses sin(phi) weights)
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    # For a constant field, average should be 1 regardless of weighting
    assert np.allclose(field_avg, 1.0, rtol=1e-10)
    
    # Now test with field = phi (should give average phi in each shell)
    field_phi = spatial_avg.phi.copy()
    r_centers, phi_avg = spatial_avg.compute_radial_average(field_phi)
    
    # Weighted average of phi over sphere should be close to pi/2
    # (integral of phi*sin(phi) from 0 to pi gives pi/2)
    overall_avg_phi = np.average(phi_avg, weights=r_centers**2)  # weight by shell volume
    assert np.allclose(overall_avg_phi, np.pi/2, rtol=0.1)


# ---------------------------------------------------------------------------
# Volume weights tests
# ---------------------------------------------------------------------------


def test_compute_volume_weights_shape(spatial_avg, mock_oper):
    """Test that volume weights have correct shape."""
    weights = spatial_avg.compute_volume_weights()
    assert weights.shape == mock_oper.X.shape


def test_compute_volume_weights_uniform(spatial_avg, mock_oper, allclose):
    """Test that volume weights are uniform for uniform grid."""
    weights = spatial_avg.compute_volume_weights()
    expected_weight = mock_oper.delta**3
    assert allclose(weights, expected_weight, rtol=1e-12)


def test_compute_volume_weights_sum(spatial_avg, mock_oper, allclose):
    """Test that sum of volume weights equals total domain volume."""
    weights = spatial_avg.compute_volume_weights()
    total_volume = np.sum(weights)
    expected_volume = mock_oper.Lx * mock_oper.Ly * mock_oper.Lz
    assert allclose(total_volume, expected_volume, rtol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_radial_average_single_bin(mock_oper):
    """Test with only one radial bin."""
    spatial_avg = SpatialAverage(mock_oper, nr=1, nrh=8, nz=8)
    field = np.ones_like(spatial_avg.X) * 3.14
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    assert len(field_avg) == 1
    assert np.allclose(field_avg[0], 3.14, rtol=1e-10)


def test_azimuthal_average_single_bin(mock_oper):
    """Test with only one azimuthal bin."""
    spatial_avg = SpatialAverage(mock_oper, nr=10, nrh=1, nz=1)
    field = np.ones_like(spatial_avg.X) * 2.71
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    assert field_avg.shape == (1, 1)
    assert np.allclose(field_avg[0, 0], 2.71, rtol=1e-10)


def test_radial_average_large_nr(mock_oper):
    """Test with more bins than grid points (some bins will be empty)."""
    spatial_avg = SpatialAverage(mock_oper, nr=100, nrh=8, nz=8)
    field = np.ones_like(spatial_avg.X)
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    
    # Non-empty bins should have value close to 1
    # Empty bins should be 0 (as set by the code)
    assert np.all((field_avg == 0) | (np.abs(field_avg - 1.0) < 0.1))


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------


def test_radial_azimuthal_consistency(spatial_avg):
    """Test that radial and azimuthal averages are consistent for spherically symmetric fields.
    
    For a spherically symmetric field f(r), both averages should give similar results
    when compared at the same radius r = sqrt(rho^2 + z^2).
    """
    # Spherically symmetric field: f(r) = r
    field = spatial_avg.r.copy()
    
    # Radial average
    r_centers, field_avg_radial = spatial_avg.compute_radial_average(field)
    
    # Azimuthal average
    rho_centers, z_centers, field_avg_azim = spatial_avg.compute_azimuthal_average(field)
    
    # For spherically symmetric field, azimuthal average should depend only on r = sqrt(rho^2 + z^2)
    # Compute r for each (rho, z) bin
    RHO, Z = np.meshgrid(rho_centers, z_centers, indexing='ij')
    r_azim = np.sqrt(RHO**2 + Z**2)
    
    # Test correlation instead of exact match
    # Flatten both arrays for correlation
    correlation = np.corrcoef(field_avg_azim.ravel(), r_azim.ravel())[0, 1]
    assert correlation > 0.95, f"Correlation {correlation} too low"
    
    # Test that most values are reasonably close
    # (discretization causes larger errors near boundaries)
    relative_error = np.abs(field_avg_azim - r_azim) / (r_azim + 1e-10)
    # At least 70% of bins should have < 30% error
    fraction_good = np.sum(relative_error < 0.3) / relative_error.size
    assert fraction_good > 0.7, f"Only {fraction_good*100:.1f}% of bins are close"


def test_linearity_radial_average(spatial_avg, allclose):
    """Test linearity: avg(a*f1 + b*f2) = a*avg(f1) + b*avg(f2)."""
    field1 = np.random.randn(*spatial_avg.X.shape)
    field2 = np.random.randn(*spatial_avg.X.shape)
    a, b = 2.5, -1.3
    
    r_centers, avg1 = spatial_avg.compute_radial_average(field1)
    _, avg2 = spatial_avg.compute_radial_average(field2)
    _, avg_combined = spatial_avg.compute_radial_average(a * field1 + b * field2)
    
    expected = a * avg1 + b * avg2
    assert allclose(avg_combined, expected, rtol=1e-10)


def test_linearity_azimuthal_average(spatial_avg, allclose):
    """Test linearity for azimuthal average."""
    field1 = np.random.randn(*spatial_avg.X.shape)
    field2 = np.random.randn(*spatial_avg.X.shape)
    a, b = 3.2, 0.7
    
    rho_centers, z_centers, avg1 = spatial_avg.compute_azimuthal_average(field1)
    _, _, avg2 = spatial_avg.compute_azimuthal_average(field2)
    _, _, avg_combined = spatial_avg.compute_azimuthal_average(a * field1 + b * field2)
    
    expected = a * avg1 + b * avg2
    assert allclose(avg_combined, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Additional physical tests
# ---------------------------------------------------------------------------


def test_radial_average_preserves_integral(spatial_avg):
    """Test that radial averaging preserves the volume integral.
    
    ∫∫∫ f dV = ∫ <f>_Omega(r) × 4πr² dr
    """
    # Random field
    field = np.random.randn(*spatial_avg.X.shape) + 5.0  # offset to be positive
    
    # Direct volume integral
    weights = spatial_avg.compute_volume_weights()
    integral_direct = np.sum(field * weights)
    
    # Radial average integral
    r_centers, field_avg = spatial_avg.compute_radial_average(field)
    dr = np.diff(spatial_avg.r_bins)  # width of each bin
    shell_volumes = 4 * np.pi * r_centers**2 * dr
    integral_radial = np.sum(field_avg * shell_volumes)
    
    # They should be approximately equal
    # (discretization causes some error)
    assert np.allclose(integral_radial, integral_direct, rtol=0.15)


def test_azimuthal_theta_independence(spatial_avg):
    """Test that azimuthal average eliminates theta dependence.
    
    A field that depends only on theta should average to zero
    (or constant if it has a mean).
    """
    # Field = cos(theta) where theta = atan2(y, x)
    theta = np.arctan2(spatial_avg.Y, spatial_avg.X)
    field = np.cos(theta)
    
    # Azimuthal average should be close to zero
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(field)
    
    # Mean over the circle should be very small
    assert np.allclose(field_avg, 0.0, atol=0.2)
