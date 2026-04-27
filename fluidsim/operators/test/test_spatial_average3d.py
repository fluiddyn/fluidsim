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
        z1d = np.linspace(-Lz / 2, Lz / 2, nz, endpoint=False)

        self.Z, self.Y, self.X = np.meshgrid(z1d, y1d, x1d, indexing="ij")
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
    assert hasattr(spatial_avg, "r")
    assert hasattr(spatial_avg, "rho")
    assert hasattr(spatial_avg, "phi")


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

    assert allclose(field_avg, 5.0, rtol=1e-10)


def test_radial_average_radial_field(spatial_avg):
    """Test radial average of f(r) = r.

    The average of r over a spherical shell is approximately r_center,
    but with discretization error. We test correlation instead of exact match.
    """
    field = spatial_avg.r.copy()
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert np.all(np.diff(field_avg) > 0), "Average should increase with radius"

    correlation = np.corrcoef(field_avg, r_centers)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"

    ratio = field_avg[1:] / r_centers[1:]
    assert np.allclose(ratio, 1.0, rtol=0.2), "Ratio should be close to 1"


def test_radial_average_quadratic_field(spatial_avg):
    """Test radial average of f(r) = r^2.

    Similar to linear case, we test correlation and approximate scaling.
    """
    field = spatial_avg.r**2
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    expected = r_centers**2

    correlation = np.corrcoef(field_avg, expected)[0, 1]
    assert correlation > 0.99, f"Correlation {correlation} too low"

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
    assert np.all(field_std >= 0)


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
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    assert allclose(field_avg, 7.0, rtol=1e-10)


@pytest.mark.parametrize("azimut_var", ["z", "rho"])
def test_azimuthal_average_z_dependent(spatial_avg, azimut_var):
    """Test azimuthal average of f(z) = z and f(rho) = rho.

    We compute the expected average by explicitly averaging the actual z values
    in each bin, accounting for the discrete grid.
    """
    if azimut_var == "z":
        field = spatial_avg.Z.copy()
        var_flat = spatial_avg.Z.ravel()
    elif azimut_var == "rho":
        field = spatial_avg.rho.copy()
        var_flat = spatial_avg.rho.ravel()

    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    expected_avg = np.zeros((spatial_avg.nrh, spatial_avg.nz))
    counts = np.zeros((spatial_avg.nrh, spatial_avg.nz))

    rho_idx_flat = spatial_avg.rho_indices.ravel()
    z_idx_flat = spatial_avg.z_indices.ravel()

    for i, var_val in enumerate(var_flat):
        irho = rho_idx_flat[i]
        iz = z_idx_flat[i]
        expected_avg[irho, iz] += var_val
        counts[irho, iz] += 1

    mask = counts > 0
    expected_avg[mask] /= counts[mask]

    np.testing.assert_allclose(
        field_avg,
        expected_avg,
        rtol=1e-12,
        err_msg="Azimuthal average should exactly match manual binning",
    )


def test_azimuthal_average_vector_field(spatial_avg):
    """Test azimuthal average of a 3D vector field."""
    shape = spatial_avg.X.shape
    vx = np.ones(shape) * 1.5
    vy = np.ones(shape) * 2.5
    vz = np.ones(shape) * 3.5
    vector_field = np.array([vx, vy, vz])

    rho_centers, z_centers, v_avg = spatial_avg.compute_azimuthal_average(
        vector_field
    )

    assert v_avg.shape == (3, spatial_avg.nrh, spatial_avg.nz)
    assert np.allclose(v_avg[0], 1.5, rtol=1e-10)
    assert np.allclose(v_avg[1], 2.5, rtol=1e-10)
    assert np.allclose(v_avg[2], 3.5, rtol=1e-10)


def test_azimuthal_average_with_std(spatial_avg):
    """Test that standard deviation is returned when requested."""
    field = np.random.randn(*spatial_avg.X.shape)
    rho_centers, z_centers, field_avg, field_std = (
        spatial_avg.compute_azimuthal_average(field, return_std=True)
    )

    assert field_avg.shape == (spatial_avg.nrh, spatial_avg.nz)
    assert field_std.shape == (spatial_avg.nrh, spatial_avg.nz)
    assert np.all(field_std >= 0)


def test_azimuthal_average_zero_field(spatial_avg, allclose):
    """Azimuthal average of zero field should be zero."""
    field = np.zeros_like(spatial_avg.X)
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    assert allclose(field_avg, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Physics tests: sin(phi) weighting
# ---------------------------------------------------------------------------


def test_radial_average_sin_phi_weighting(spatial_avg):
    """Test that sin(phi) weighting is correctly applied. Should be pi/2, with a small bias due to asymetry of the box."""
    field = np.ones_like(spatial_avg.X)

    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert np.allclose(field_avg, 1.0, rtol=1e-10)

    field_phi = spatial_avg.phi.copy()
    r_centers, phi_avg = spatial_avg.compute_radial_average(field_phi)

    overall_avg_phi = np.average(phi_avg)
    assert np.allclose(overall_avg_phi, np.pi / 2, rtol=0.1)


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
    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    assert field_avg.shape == (1, 1)
    assert np.allclose(field_avg[0, 0], 2.71, rtol=1e-10)


def test_radial_average_large_nr(mock_oper):
    """Test with more bins than grid points (some bins will be empty)."""
    spatial_avg = SpatialAverage(mock_oper, nr=100, nrh=8, nz=8)
    field = np.ones_like(spatial_avg.X)
    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    assert np.all((field_avg == 0) | (np.abs(field_avg - 1.0) < 0.1))


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------


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
    _, _, avg_combined = spatial_avg.compute_azimuthal_average(
        a * field1 + b * field2
    )

    expected = a * avg1 + b * avg2
    assert allclose(avg_combined, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Additional physical tests
# ---------------------------------------------------------------------------

# HERE !!!!!!!!!!!!


def test_radial_average_manual_reconstruction(spatial_avg):
    """Test that we can reconstruct the field sum from radial averages.

    The radial average with sin(phi) weights represents:
    <f>_Omega(r) = (sum f_i * sin(phi_i)) / (sum sin(phi_i))

    We can reconstruct the weighted sum:
    sum(f_i * sin(phi_i)) = <f>_Omega(r) * sum(sin(phi_i))
    """
    np.random.seed(42)
    field = np.random.randn(*spatial_avg.X.shape)

    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    reconstructed_weighted_sum = 0.0
    expected_weighted_sum = 0.0

    for i_bin in range(spatial_avg.nr):
        mask = spatial_avg.r_indices == i_bin

        expected_weighted_sum += np.sum(
            field[mask] * np.sin(spatial_avg.phi[mask])
        )

        sum_sin_phi = np.sum(np.sin(spatial_avg.phi[mask]))
        reconstructed_weighted_sum += field_avg[i_bin] * sum_sin_phi

    np.testing.assert_allclose(
        reconstructed_weighted_sum,
        expected_weighted_sum,
        rtol=1e-12,
        err_msg="Reconstructed weighted sum should match direct calculation",
    )


def test_radial_average_volume_weighted_correctly(spatial_avg):
    """Test that radial average preserves the volume-weighted sum when
    we account for the sin(phi) weighting correctly.

    The key: field_avg is computed with sin(phi) weights, so to get back
    the volume integral, we need to account for how sin(phi) relates to volume.
    """
    np.random.seed(42)
    field = np.random.randn(*spatial_avg.X.shape)

    weights = spatial_avg.compute_volume_weights()
    integral_direct = np.sum(field * weights)

    integral_reconstructed = 0.0

    for i_bin in range(spatial_avg.nr):
        mask = spatial_avg.r_indices == i_bin

        if not np.any(mask):
            continue

        integral_reconstructed += np.sum(field[mask] * weights[mask])

    np.testing.assert_allclose(
        integral_reconstructed,
        integral_direct,
        rtol=1e-12,
        err_msg="Sum over bins should equal total sum",
    )


def test_radial_average_preserves_weighted_sum(spatial_avg):
    """Test that radial averaging correctly computes sin(phi)-weighted averages.

    We verify that the average can be used to reconstruct the sin(phi)-weighted sum.
    """
    np.random.seed(42)
    field = np.random.randn(*spatial_avg.X.shape)

    r_centers, field_avg = spatial_avg.compute_radial_average(field)

    for i_bin in range(spatial_avg.nr):
        mask = spatial_avg.r_indices == i_bin

        if not np.any(mask):
            assert field_avg[i_bin] == 0.0
            continue

        sum_f_weighted = np.sum(field[mask] * np.sin(spatial_avg.phi[mask]))
        sum_weights = np.sum(np.sin(spatial_avg.phi[mask]))
        expected_avg = sum_f_weighted / sum_weights if sum_weights > 0 else 0.0

        np.testing.assert_allclose(
            field_avg[i_bin],
            expected_avg,
            rtol=1e-12,
            err_msg=f"Bin {i_bin} average should match direct calculation",
        )


def test_azimuthal_average_preserves_sum(spatial_avg):
    """Test that azimuthal averaging correctly computes unweighted averages.

    For azimuthal average, we can exactly reconstruct the total sum.
    """
    np.random.seed(42)
    field = np.random.randn(*spatial_avg.X.shape)

    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    sum_direct = np.sum(field)

    sum_reconstructed = 0.0

    for irho in range(spatial_avg.nrh):
        for iz in range(spatial_avg.nz):
            mask = (spatial_avg.rho_indices == irho) & (
                spatial_avg.z_indices == iz
            )

            n_points = np.sum(mask)
            if n_points > 0:
                sum_reconstructed += field_avg[irho, iz] * n_points

    np.testing.assert_allclose(
        sum_reconstructed,
        sum_direct,
        rtol=1e-12,
        err_msg="Reconstructed sum should match direct sum",
    )


def test_azimuthal_theta_independence(spatial_avg):
    """Test that azimuthal average eliminates theta dependence.

    For f(theta) = cos(theta), the azimuthal average should be:
    <f>_theta(rho, z) = (1 / N_theta) × sum of cos(theta_i) for points at (rho, z)

    On a Cartesian grid, points are NOT uniformly distributed in theta,
    so we compute the expected average accounting for the actual theta distribution.
    """
    theta = np.arctan2(spatial_avg.Y, spatial_avg.X)
    field = np.cos(theta)

    rho_centers, z_centers, field_avg = spatial_avg.compute_azimuthal_average(
        field
    )

    expected_avg = np.zeros((spatial_avg.nrh, spatial_avg.nz))
    counts = np.zeros((spatial_avg.nrh, spatial_avg.nz))

    field_flat = field.ravel()
    rho_idx_flat = spatial_avg.rho_indices.ravel()
    z_idx_flat = spatial_avg.z_indices.ravel()

    for i, f_val in enumerate(field_flat):
        irho = rho_idx_flat[i]
        iz = z_idx_flat[i]
        expected_avg[irho, iz] += f_val
        counts[irho, iz] += 1

    mask = counts > 0
    expected_avg[mask] /= counts[mask]

    np.testing.assert_allclose(
        field_avg,
        expected_avg,
        rtol=1e-12,
        err_msg="Azimuthal average should match manual binning",
    )
