"""Spatial radial and azimuthal average in 3D (:mod:`fluidsim.operator.spatial_average3d`)
==========================================================================================

Provides:

.. autoclass:: SpatialAverage
   :members:
   :private-members:

"""

import numpy as np

from transonic import boost

from fluiddyn.util import mpi

from fluidsim.operators.coord_system3d import CoordSystem3DConverter

Af3d = "float64[:,:,:]"
Af1d = "float64[]"


@boost
def loop_spherical(arr_r0r1r2: Af3d, rs: Af1d, R2: Af3d):
    """Compute the radial field summed over theta and phi."""
    deltars = rs[1]
    nrs = len(rs)
    _field_spherical_sum = np.zeros(nrs)
    nr0, nr1, nr2 = arr_r0r1r2.shape
    for ir0 in range(nr0):
        for ir1 in range(nr1):
            for ir2 in range(nr2):
                value = arr_r0r1r2[ir0, ir1, ir2]
                kappa = np.sqrt(R2[ir0, ir1, ir2])
                ir = int(kappa / deltars)
                if ir >= nrs - 1:
                    ir = nrs - 1
                    _field_spherical_sum[ir] += value
                else:
                    coef_share = (kappa - rs[ir]) / deltars
                    _field_spherical_sum[ir] += (1 - coef_share) * value
                    _field_spherical_sum[ir + 1] += coef_share * value

    return _field_spherical_sum


@boost
def loop_azimuthal_rhrz(
    field_to_avg: Af3d,
    rho_list: Af1d,
    rho_field: Af3d,
    z_list: Af1d,
    z_field: Af3d,
):
    """Compute the _z-rh field summed over theta."""
    _deltarho = rho_list[1]
    _deltaz = z_list[1] - z_list[0]
    _z_min = z_list[0]
    _nrho = len(rho_list)
    _nz = len(z_list)
    _field_circular_sum = np.zeros((_nz, _nrho))
    n0, n1, n2 = field_to_avg.shape
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                value = field_to_avg[i0, i1, i2]
                kappa = rho_field[i0, i1, i2]
                irho = int(kappa / _deltarho)
                _z = z_field[i0, i1, i2]
                iz = int(round((_z - _z_min) / _deltaz))
                if iz >= _nz - 1:
                    iz = _nz - 1
                if irho >= _nrho - 1:
                    irho = _nrho - 1
                    _field_circular_sum[iz, irho] += value
                else:
                    coef_share = (kappa - rho_list[irho]) / _deltarho
                    _field_circular_sum[iz, irho] += (1 - coef_share) * value
                    _field_circular_sum[iz, irho + 1] += coef_share * value
    return _field_circular_sum


class SpatialAverage:
    r"""Compute spatial average on a field.

    Notes
    -----

    .. |dtheta| mathmacro:: \mathop{d\theta}
    .. |dphi| mathmacro:: \mathop{d\phi}
    .. |dx| mathmacro:: \mathop{dx}


    The continuous form of the radial average of field :math:`f` over a sphere :math:`\Omega` of radius :math:`r` with is:

    .. math:: \langle f \rangle_{\Omega}^{continuous}(r) = 1/(4\pi) \int f(r,\theta,\phi) sin(\phi) \dtheta \dphi.

    Its discrete form in a 3D domain of uniform cubic cells of sizes :math:`\mathop{dx} \times \mathop{dy} \times \mathop{dz}` is approximated at radius :math:`r` as:

    .. math:: \langle f \rangle_{\Omega}^{discrete}(r) = \frac{\text{Sum}_f(r)}{\text{Weight}_f(r)},

    with

    .. math:: \text{Sum}_f(r) = \sum_{\substack{i,j,k \\ r \leq r_i < r + \Delta r \\ r - \Delta r \leq r_{i-1} < r}} f_{i,j,k}(r_i,\theta_j,\phi_k)\left(1 - \frac{r_i - r}{\Delta r}\right) + f_{i-1,j,k}(r_{i-1},\theta_j,\phi_k)\frac{r_{i-1} - r}{\Delta r},

    and

    .. math:: \text{Weight}_f(r) = \sum_{\substack{i,j,k \\ r \leq r_i < r + \Delta r \\ r - \Delta r \leq r_{i-1} < r}} \mathbf{1}_{i,j,k}(r_i,\theta_j,\phi_k) \left(1 - \frac{r_i - r}{\Delta r}\right) + \mathbf{1}_{i-1,j,k}(r_{i-1},\theta_j,\phi_k)\frac{r_{i-1} - r}{\Delta r}.

    Where :math:`\mathbf{1}` is a 3D field of values :math:`1` and :math:`\Delta r = \mathop{dr}\text{min}(\mathop{dx}, \mathop{dy}, \mathop{dz})` is the radius step (worth :math:`\mathop{dx} = \mathop{dy} = \mathop{dz}` by default).

    The continuous form of the azimuthal average of field :math:`f` over circles of radius :math:`\rho` at each height :math:`z` is:

    .. math:: \langle f \rangle_{\theta}(\rho, z) = 1/(2\pi) \int f(\rho,\theta,z) \mathop{d\theta}.

    Its discrete form in a 3D domain of uniform cubic cells is approximated as:

    .. math:: \langle f \rangle_{\theta}^{discrete}(\rho, z) = \frac{\text{Sum}_f(\rho, z)}{\text{Weight}_f(\rho, z)},

    with

    .. math:: \text{Sum}_f(\rho, z) = \sum_{\substack{i,j,k \\ z \leq z_k < z + \Delta z \\ \rho \leq \rho_i < \rho + \Delta \rho \\ \rho - \Delta \rho \leq \rho_{i-1} < \rho}} f_{i,j,k}(\rho_i,\theta_j,z_k)\left(1 - \frac{\rho_i - \rho}{\Delta \rho}\right) + f_{i-1,j,k}(\rho_{i-1},\theta_j,z_k)\frac{\rho_{i-1} - \rho}{\Delta \rho},

    and

    .. math:: \text{Weight}_f(\rho, z) = \sum_{\substack{i,j,k \\ z \leq z_k < z + \Delta z \\ \rho \leq r_i < \rho + \Delta \rho \\ \rho - \Delta \rho \leq \rho_{i-1} < \rho}} \mathbf{1}_{i,j,k}(\rho_i,\theta_j,z_k) \left(1 - \frac{\rho_i - \rho}{\Delta \rho}\right) + \mathbf{1}_{i-1,j,k}(\rho_{i-1},\theta_j,z_k)\frac{\rho_{i-1} - \rho}{\Delta \rho}.

    Where :math:`\Delta \rho = \mathop{drh}\text{min}(\mathop{dx}, \mathop{dy}, \mathop{dz})` and :math:`\Delta z = \widetilde{\mathop{dz}}\text{min}(\mathop{dx}, \mathop{dy}, \mathop{dz})` (:math:`\widetilde{\mathop{dz}}` is an input parameter, ratio of the required step by the step from the grid) the horizontal radius and vertical steps (both worth :math:`\mathop{dx} = \mathop{dy} = \mathop{dz}` by default).

    Parameters
    ----------
    oper : Operator
        Operator object containing the mesh information (X, Y, Z coordinates).
        The mesh is assumed uniform with the same grid spacing dx=dy=dz in all
        directions, but the domain lengths Lx, Ly, Lz (and therefore the number
        of grid points Nx, Ny, Nz) can differ.
    dr : float, optional
        Radial bin size factor (default: 2.0)
    drh : float, optional
        Azimuthal rho bin size factor (default: 2.0)
    dz : float, optional
        Azimuthal z bin size factor (default: 1.0)
    shift_origin : bool, optional
        If True, shift origin to domain center (default: True)
    """

    def __init__(self, oper, dr=1.0, drh=1.0, dz=1.0, shift_origin=True):
        self.oper = oper

        # Compute bin spacings
        delta_min = min(oper.Lx / oper.nx, oper.Ly / oper.ny, oper.Lz / oper.nz)
        self.deltar = delta_min * dr
        self.deltarh = delta_min * drh
        self.deltaz = delta_min * dz

        # Get local Cartesian coordinates from the operator
        X, Y, Z = oper.get_XYZ_loc()

        # Get domain sizes
        Lx = oper.Lx
        Ly = oper.Ly
        Lz = oper.Lz

        # Initialize coordinate converter with shift option
        self.coord_conv = CoordSystem3DConverter(
            X, Y, Z, Lx, Ly, Lz, shift_origin=shift_origin
        )

        # Store coordinates (already shifted if shift_origin=True)
        self.X = self.coord_conv.x
        self.Y = self.coord_conv.y
        self.Z = self.coord_conv.z

        # Compute cylindrical and spherical coordinates
        self._compute_coordinates()

        # Prepare bins for averaging
        self._prepare_radial_bins()
        self._prepare_azimuthal_bins()

        # Compute weights (total counts in each bin)
        self._compute_weights()

    def _compute_coordinates(self):
        r"""Compute cylindrical and spherical coordinate arrays

        Notes
        -----

        Uses CoordSystem3DConverter to compute :math:`\rho` and :math:`r`.
        """
        self.rho = self.coord_conv.rh
        self.r = self.coord_conv.r_not0

    def _prepare_radial_bins(self):
        r"""Prepare bins for radial averaging.

        Notes
        -----

        Creates uniformly spaced bin at :math:`\Delta r = \mathop{dr}\text{min}(\mathop{dx}, \mathop{dy}, \mathop{dz})` centers spanning [r_min, r_max] globally.
        """
        # Find local min/max
        r_min_loc = np.min(self.r)
        r_max_loc = np.max(self.r)

        # Gather global min/max across all processes
        if mpi.nb_proc > 1:
            r_min_all = mpi.comm.gather(r_min_loc, root=0)
            r_max_all = mpi.comm.gather(r_max_loc, root=0)

            if mpi.rank == 0:
                r_min = np.min(r_min_all)
                r_max = np.max(r_max_all)
            else:
                r_min = None
                r_max = None

            # Broadcast to all processes
            r_min = mpi.comm.bcast(r_min, root=0)
            r_max = mpi.comm.bcast(r_max, root=0)
        else:
            r_min = r_min_loc
            r_max = r_max_loc

        self.nr = int((r_max - r_min) / self.deltar) + 1

        # Create uniformly spaced centers
        self.r_centers = np.linspace(
            r_min, r_min + self.nr * self.deltar, self.nr, endpoint=False
        )

    def _prepare_azimuthal_bins(self):
        r"""Prepare bins for azimuthal averaging

        Notes
        -----

        Creates uniformly spaced bin centers for :math:`\rho` and :math:`z`.
        """
        rho_max_loc = np.max(self.rho)
        rho_min_loc = np.min(self.rho)
        z_min_loc = np.min(self.Z)
        z_max_loc = np.max(self.Z)

        if mpi.nb_proc > 1:
            rho_max_all = mpi.comm.gather(rho_max_loc, root=0)
            rho_min_all = mpi.comm.gather(rho_min_loc, root=0)
            z_min_all = mpi.comm.gather(z_min_loc, root=0)
            z_max_all = mpi.comm.gather(z_max_loc, root=0)

            if mpi.rank == 0:
                rho_max = np.max(rho_max_all)
                rho_min = np.max(rho_min_all)
                z_min = np.min(z_min_all)
                z_max = np.max(z_max_all)
            else:
                rho_max = None
                rho_min = None
                z_min = None
                z_max = None

            rho_max = mpi.comm.bcast(rho_max, root=0)
            rho_min = mpi.comm.bcast(rho_min, root=0)
            z_min = mpi.comm.bcast(z_min, root=0)
            z_max = mpi.comm.bcast(z_max, root=0)
        else:
            rho_max = rho_max_loc
            rho_min = rho_min_loc
            z_min = z_min_loc
            z_max = z_max_loc

        # Create uniform bin centers
        self.nrh = int((rho_max - rho_min) / self.deltarh) + 1
        self._nz = int((z_max - z_min) / self.deltaz) + 1

        self.rho_centers = np.linspace(
            rho_min, rho_min + self.nrh * self.deltarh, self.nrh, endpoint=False
        )
        self.z_centers = np.linspace(
            z_min, z_min + self._nz * self.deltaz, self._nz, endpoint=False
        )

    def _compute_weights(self):
        r"""Compute the total weight (count) in each bin across all processes.

        Notes
        -----

        This computes the normalization factor for averaging.
        For radial average: counts points in each spherical shell.
        For azimuthal average: counts points in each :math:`(\rho, z)` bin.
        """
        ones_field = np.ones_like(self.X)

        radial_weights_loc = loop_spherical(ones_field, self.r_centers, self.r**2)

        azimuthal_weights_loc = loop_azimuthal_rhrz(
            ones_field, self.rho_centers, self.rho, self.z_centers, self.Z
        )

        # Sum across MPI processes
        if mpi.nb_proc > 1:
            radial_all = mpi.comm.gather(radial_weights_loc, root=0)
            if mpi.rank == 0:
                self.radial_weights = np.sum(radial_all, axis=0)
            else:
                self.radial_weights = None
            self.radial_weights = mpi.comm.bcast(self.radial_weights, root=0)

            azimuthal_all = mpi.comm.gather(azimuthal_weights_loc, root=0)
            if mpi.rank == 0:
                self.azimuthal_weights = np.sum(azimuthal_all, axis=0)
            else:
                self.azimuthal_weights = None
            self.azimuthal_weights = mpi.comm.bcast(
                self.azimuthal_weights, root=0
            )
        else:
            self.radial_weights = radial_weights_loc
            self.azimuthal_weights = azimuthal_weights_loc

    # ------------------------------------------------------------------ #
    #  Radial average  <f>_Omega(r)                                        #
    # ------------------------------------------------------------------ #

    def compute_radial_average(self, field, return_std=False):
        """Compute the solid-angle average of a scalar or vector field

        Parameters
        ----------
        field : array_like
            Scalar field of shape (Nx_loc, Ny_loc, Nz_loc) or
            vector field of shape (3, Nx_loc, Ny_loc, Nz_loc) in the spherical basis.
            Each MPI process provides its local slice.
        return_std : bool, optional
            If True, also return the standard deviation (default: False).

        Returns
        -------
        r_centers : ndarray, shape (nr,)
            Radial bin centres (same on all processes).
        field_avg : ndarray, shape (nr,) or (3, nr)
            Solid-angle average in each bin (same on all processes).
        field_std : ndarray, same shape as field_avg (only if return_std=True)
            Standard deviation in each bin (same on all processes).
        """
        is_vector = np.ndim(field) == 4 and np.shape(field)[0] == 3

        if is_vector:
            field_avg = np.zeros((3, self.nr))
            field_std = np.zeros((3, self.nr)) if return_std else None
            for i in range(3):
                out = self._radial_average_scalar(field[i], return_std)
                if return_std:
                    field_avg[i], field_std[i] = out
                else:
                    field_avg[i] = out
        else:
            out = self._radial_average_scalar(field, return_std)
            if return_std:
                field_avg, field_std = out
            else:
                field_avg = out

        if return_std:
            return self.r_centers, field_avg, field_std
        return self.r_centers, field_avg

    def _radial_average_scalar(self, field, return_std=False):
        """Average over radial bins for a scalar field using loop_spherical.

        Parameters
        ----------
        field : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Local field slice on this process.
        return_std : bool

        Returns
        -------
        field_avg : ndarray, shape (nr,)
            Global average across all processes.
        field_std : ndarray, shape (nr,) — only if return_std is True
        """
        # Local sum of field in each bin
        sum_f_loc = loop_spherical(field, self.r_centers, self.r**2)

        # MPI reduction
        if mpi.nb_proc > 1:
            sum_f_all = mpi.comm.gather(sum_f_loc, root=0)

            if mpi.rank == 0:
                sum_f = np.sum(sum_f_all, axis=0)
            else:
                sum_f = None

            sum_f = mpi.comm.bcast(sum_f, root=0)
        else:
            sum_f = sum_f_loc

        # Compute average
        mask_nonzero = self.radial_weights > 0
        field_avg = np.zeros(self.nr)
        field_avg[mask_nonzero] = (
            sum_f[mask_nonzero] / self.radial_weights[mask_nonzero]
        )

        if not return_std:
            return field_avg

        # Compute variance and std
        sum_f2_loc = loop_spherical(field**2, self.r_centers, self.r**2)

        if mpi.nb_proc > 1:
            sum_f2_all = mpi.comm.gather(sum_f2_loc, root=0)

            if mpi.rank == 0:
                sum_f2 = np.sum(sum_f2_all, axis=0)
            else:
                sum_f2 = None

            sum_f2 = mpi.comm.bcast(sum_f2, root=0)
        else:
            sum_f2 = sum_f2_loc

        f2_avg = np.zeros(self.nr)
        f2_avg[mask_nonzero] = (
            sum_f2[mask_nonzero] / self.radial_weights[mask_nonzero]
        )
        field_var = np.maximum(f2_avg - field_avg**2, 0.0)
        field_std = np.sqrt(field_var)

        return field_avg, field_std

    # ------------------------------------------------------------------ #
    #  Azimuthal average  <f>_theta(rho, z)                               #
    # ------------------------------------------------------------------ #

    def compute_azimuthal_average(self, field, return_std=False):
        """Compute the azimuthal average of a scalar or vector field

        Parameters
        ----------
        field : array_like
            Scalar field of shape (Nx_loc, Ny_loc, Nz_loc) or
            vector field of shape (3, Nx_loc, Ny_loc, Nz_loc) in the cylindrical basis.
            Each MPI process provides its local slice.
        return_std : bool, optional
            If True, also return the standard deviation (default: False).

        Returns
        -------
        rho_centers : ndarray, shape (nrh,)
            Bin centres in rho (same on all processes).
        z_centers : ndarray, shape (nz,)
            Bin centres in z (same on all processes).
        field_avg : ndarray, shape (nz, nrh) or (3, nz, nrh)
            Azimuthal average in each (rho, z) bin (same on all processes).
        field_std : ndarray, same shape as field_avg (only if return_std=True)
        """
        is_vector = np.ndim(field) == 4 and np.shape(field)[0] == 3

        if is_vector:
            field_avg = np.zeros((3, self._nz, self.nrh))
            field_std = np.zeros((3, self._nz, self.nrh)) if return_std else None
            for i in range(3):
                out = self._azimuthal_average_scalar(field[i], return_std)
                if return_std:
                    field_avg[i], field_std[i] = out
                else:
                    field_avg[i] = out
        else:
            out = self._azimuthal_average_scalar(field, return_std)
            if return_std:
                field_avg, field_std = out
            else:
                field_avg = out

        if return_std:
            return self.rho_centers, self.z_centers, field_avg, field_std
        return self.rho_centers, self.z_centers, field_avg

    def _azimuthal_average_scalar(self, field, return_std=False):
        r"""Average over :math:`(\rho, z)` bins for a scalar field using loop_azimuthal_rhrz.

        Parameters
        ----------
        field : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Local field slice on this process.
        return_std : bool

        Returns
        -------
        field_avg : ndarray, shape (nz, nrh)
            Global average across all processes.
        field_std : ndarray, shape (nz, nrh) — only if return_std is True
        """
        # Local sum of field in each (rho, z) bin
        sum_f_loc = loop_azimuthal_rhrz(
            field, self.rho_centers, self.rho, self.z_centers, self.Z
        )

        # MPI reduction
        if mpi.nb_proc > 1:
            sum_f_all = mpi.comm.gather(sum_f_loc, root=0)

            if mpi.rank == 0:
                sum_f = np.sum(sum_f_all, axis=0)
            else:
                sum_f = None

            sum_f = mpi.comm.bcast(sum_f, root=0)
        else:
            sum_f = sum_f_loc

        # Compute average
        mask_nonzero = self.azimuthal_weights > 0
        field_avg = np.zeros((self._nz, self.nrh))
        field_avg[mask_nonzero] = (
            sum_f[mask_nonzero] / self.azimuthal_weights[mask_nonzero]
        )

        if not return_std:
            return field_avg

        # Compute variance and std
        sum_f2_loc = loop_azimuthal_rhrz(
            field**2, self.rho_centers, self.rho, self.z_centers, self.Z
        )

        if mpi.nb_proc > 1:
            sum_f2_all = mpi.comm.gather(sum_f2_loc, root=0)

            if mpi.rank == 0:
                sum_f2 = np.sum(sum_f2_all, axis=0)
            else:
                sum_f2 = None

            sum_f2 = mpi.comm.bcast(sum_f2, root=0)
        else:
            sum_f2 = sum_f2_loc

        f2_avg = np.zeros((self._nz, self.nrh))
        f2_avg[mask_nonzero] = (
            sum_f2[mask_nonzero] / self.azimuthal_weights[mask_nonzero]
        )
        field_var = np.maximum(f2_avg - field_avg**2, 0.0)
        field_std = np.sqrt(field_var)

        return field_avg, field_std

    def compute_volume_weights(self):
        r"""Compute the volume weight of each mesh cell

        Notes
        -----

        For a uniform isotropic mesh :math:`(\mathop{dx} = \mathop{dy} = \mathop{dz} = \mathop{d})`, all cells have the same
        volume :math:`\mathop{d}^3`. The grid spacing is read from the operator if available.

        Returns
        -------
        Volumes : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Volume weights on this process.
        Volumes : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Volume weights on this process.
        """
        d = self.oper.delta if hasattr(self.oper, "delta") else 1.0
        return np.full_like(self.X, d**3)
