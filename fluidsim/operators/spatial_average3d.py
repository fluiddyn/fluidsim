"""Spatial radial and azimuthal average in 3D (:mod:`fluidsim.operator.spatial_average3d`)
==========================================================================================

Provides:

.. autoclass:: SpatialAverage
   :members:
   :private-members:

"""

import numpy as np
from fluiddyn.util import mpi

from fluidsim.operators.coord_system3d import CoordSystem3DConverter


class SpatialAverage:
    """Compute spatial average on a field

    The radial average averages over a sphere of radius r (solid angle average):

        <f>_Omega(r) = 1/(4*pi) * integral f(r,theta,phi) sin(phi) dtheta dphi

    The azimuthal average averages over circles of radius rho at each height z:

        <f>_theta(rho,z) = 1/(2*pi) * integral f(rho,theta,z) dtheta

    Parameters
    ----------
    oper : Operator
        Operator object containing the mesh information (X, Y, Z coordinates).
        The mesh is assumed uniform with the same grid spacing dx=dy=dz in all
        directions, but the domain lengths Lx, Ly, Lz (and therefore the number
        of grid points Nx, Ny, Nz) can differ.
    nr : int, optional
        Number of radial bins for radial averaging (default: 50)
    nrh : int, optional
        Number of rho bins for azimuthal averaging (default: 50)
    nz : int, optional
        Number of z bins for azimuthal averaging (default: 50)
    """

    def __init__(self, oper, nr=50, nrh=50, nz=50):
        self.oper = oper
        self.nr = nr
        self.nrh = nrh
        self.nz = nz

        # Get local Cartesian coordinates from the operator
        X, Y, Z = oper.get_XYZ_loc()
        self.X = X
        self.Y = Y
        self.Z = Z

        # Initialize coordinate converter
        self.coord_conv = CoordSystem3DConverter(X, Y, Z)

        # Compute cylindrical and spherical coordinates
        self._compute_coordinates()

        # Prepare bins for averaging
        self._prepare_radial_bins()
        self._prepare_azimuthal_bins()

        # Precompute indices for binning on this process
        self._compute_bin_indices()

    def _compute_coordinates(self):
        """Compute cylindrical and spherical coordinate arrays

        Uses CoordSystem3DConverter to compute rho and r, as well as phi
        for the sin(phi) weighting in radial averages.
        """
        self.rho = self.coord_conv.rh
        self.r = self.coord_conv.r_not0
        self.phi = np.arccos(np.clip(self.Z / self.coord_conv.r_not0, -1.0, 1.0))

    def _prepare_radial_bins(self):
        """Prepare bins for radial averaging

        Bins span [r_min, r_max] globally across all MPI processes.
        """
        # Find local min/max
        r_min_loc = np.min(self.r[self.r > 0]) if np.any(self.r > 0) else np.inf
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

        # Create uniform bins
        self.r_bins = np.linspace(r_min, r_max, self.nr + 1)
        self.r_centers = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])

    def _prepare_azimuthal_bins(self):
        """Prepare bins for azimuthal averaging

        Two independent bin arrays are built: one for rho in [0, rho_max]
        and one for z in [z_min, z_max], determined globally across all processes.
        """
        rho_max_loc = np.max(self.rho)
        z_min_loc = np.min(self.Z)
        z_max_loc = np.max(self.Z)

        if mpi.nb_proc > 1:
            rho_max_all = mpi.comm.gather(rho_max_loc, root=0)
            z_min_all = mpi.comm.gather(z_min_loc, root=0)
            z_max_all = mpi.comm.gather(z_max_loc, root=0)

            if mpi.rank == 0:
                rho_max = np.max(rho_max_all)
                z_min = np.min(z_min_all)
                z_max = np.max(z_max_all)
            else:
                rho_max = None
                z_min = None
                z_max = None

            rho_max = mpi.comm.bcast(rho_max, root=0)
            z_min = mpi.comm.bcast(z_min, root=0)
            z_max = mpi.comm.bcast(z_max, root=0)
        else:
            rho_max = rho_max_loc
            z_min = z_min_loc
            z_max = z_max_loc

        self.rho_bins = np.linspace(0.0, rho_max, self.nrh + 1)
        self.rho_centers = 0.5 * (self.rho_bins[:-1] + self.rho_bins[1:])

        self.z_bins = np.linspace(z_min, z_max, self.nz + 1)
        self.z_centers = 0.5 * (self.z_bins[:-1] + self.z_bins[1:])

    def _compute_bin_indices(self):
        """Precompute bin indices for each mesh point on this process

        This is done once at initialization to avoid repeated digitize calls.
        """
        self.r_indices = np.clip(np.digitize(self.r, self.r_bins) - 1, 0, self.nr - 1)

        self.rho_indices = np.clip(
            np.digitize(self.rho, self.rho_bins) - 1, 0, self.nrh - 1
        )
        self.z_indices = np.clip(np.digitize(self.Z, self.z_bins) - 1, 0, self.nz - 1)

    # ------------------------------------------------------------------ #
    #  Radial average  <f>_Omega(r)                                        #
    # ------------------------------------------------------------------ #

    def compute_radial_average(self, field, return_std=False):
        """Compute the solid-angle average of a scalar or vector field

        Implements the spherical average:

            <f>_Omega(r) = 1/(4*pi) * integral f(r,theta,phi) sin(phi) dtheta dphi

        The sin(phi) factor is the geometrical weight of each mesh point on the
        unit sphere (area element on the sphere = sin(phi) dtheta dphi).

        Parameters
        ----------
        field : array_like
            Scalar field of shape (Nx_loc, Ny_loc, Nz_loc) or
            vector field of shape (3, Nx_loc, Ny_loc, Nz_loc).
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
        weights = np.sin(self.phi)

        is_vector = np.ndim(field) == 4 and np.shape(field)[0] == 3

        if is_vector:
            field_avg = np.zeros((3, self.nr))
            field_std = np.zeros((3, self.nr)) if return_std else None
            for i in range(3):
                out = self._radial_average_scalar(field[i], weights, return_std)
                if return_std:
                    field_avg[i], field_std[i] = out
                else:
                    field_avg[i] = out
        else:
            out = self._radial_average_scalar(field, weights, return_std)
            if return_std:
                field_avg, field_std = out
            else:
                field_avg = out

        if return_std:
            return self.r_centers, field_avg, field_std
        return self.r_centers, field_avg

    def _radial_average_scalar(self, field, weights, return_std=False):
        """Weighted bincount average over radial bins for a scalar field

        Computes the sin(phi)-weighted mean in each radial bin, which
        discretises the continuous integral. Uses MPI reduction to combine
        contributions from all processes.

        Parameters
        ----------
        field : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Local field slice on this process.
        weights : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            sin(phi) values on this process.
        return_std : bool

        Returns
        -------
        field_avg : ndarray, shape (nr,)
            Global average across all processes.
        field_std : ndarray, shape (nr,) — only if return_std is True
        """
        f = field.ravel()
        w = weights.ravel()
        idx = self.r_indices.ravel()

        sum_fw_loc = np.bincount(idx, weights=f * w, minlength=self.nr)
        sum_w_loc = np.bincount(idx, weights=w, minlength=self.nr)

        if mpi.nb_proc > 1:
            sum_fw_all = mpi.comm.gather(sum_fw_loc, root=0)
            sum_w_all = mpi.comm.gather(sum_w_loc, root=0)

            if mpi.rank == 0:
                sum_fw = np.sum(sum_fw_all, axis=0)
                sum_w = np.sum(sum_w_all, axis=0)
            else:
                sum_fw = None
                sum_w = None

            sum_fw = mpi.comm.bcast(sum_fw, root=0)
            sum_w = mpi.comm.bcast(sum_w, root=0)
        else:
            sum_fw = sum_fw_loc
            sum_w = sum_w_loc

        mask_nonzero = sum_w > 0

        field_avg = np.zeros(self.nr)
        field_avg[mask_nonzero] = sum_fw[mask_nonzero] / sum_w[mask_nonzero]

        if not return_std:
            return field_avg

        sum_f2w_loc = np.bincount(idx, weights=f**2 * w, minlength=self.nr)

        if mpi.nb_proc > 1:
            sum_f2w_all = mpi.comm.gather(sum_f2w_loc, root=0)

            if mpi.rank == 0:
                sum_f2w = np.sum(sum_f2w_all, axis=0)
            else:
                sum_f2w = None

            sum_f2w = mpi.comm.bcast(sum_f2w, root=0)
        else:
            sum_f2w = sum_f2w_loc

        f2_avg = np.zeros(self.nr)
        f2_avg[mask_nonzero] = sum_f2w[mask_nonzero] / sum_w[mask_nonzero]
        field_var = np.maximum(f2_avg - field_avg**2, 0.0)
        field_std = np.sqrt(field_var)

        return field_avg, field_std

    # ------------------------------------------------------------------ #
    #  Azimuthal average  <f>_theta(rho, z)                               #
    # ------------------------------------------------------------------ #

    def compute_azimuthal_average(self, field, return_std=False):
        """Compute the azimuthal average of a scalar or vector field

        Implements the azimuthal average over the angle theta:

            <f>_theta(rho, z) = 1/(2*pi) * integral f(rho, theta, z) dtheta

        All mesh points sharing the same (rho, z) bin but different theta
        contribute equally (uniform weight), which is the correct discretisation
        of the 1/(2*pi) integral over theta.

        Parameters
        ----------
        field : array_like
            Scalar field of shape (Nx_loc, Ny_loc, Nz_loc) or
            vector field of shape (3, Nx_loc, Ny_loc, Nz_loc).
            Each MPI process provides its local slice.
        return_std : bool, optional
            If True, also return the standard deviation (default: False).

        Returns
        -------
        rho_centers : ndarray, shape (nrh,)
            Bin centres in rho (same on all processes).
        z_centers : ndarray, shape (nz,)
            Bin centres in z (same on all processes).
        field_avg : ndarray, shape (nrh, nz) or (3, nrh, nz)
            Azimuthal average in each (rho, z) bin (same on all processes).
        field_std : ndarray, same shape as field_avg (only if return_std=True)
        """
        is_vector = np.ndim(field) == 4 and np.shape(field)[0] == 3

        if is_vector:
            field_avg = np.zeros((3, self.nrh, self.nz))
            field_std = np.zeros((3, self.nrh, self.nz)) if return_std else None
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
        """Uniform bincount average over (rho, z) bins for a scalar field

        Each mesh point in a (rho, z) bin contributes with weight 1,
        discretising the uniform 1/(2*pi) integral over theta.
        Uses MPI reduction to combine contributions from all processes.

        Parameters
        ----------
        field : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Local field slice on this process.
        return_std : bool

        Returns
        -------
        field_avg : ndarray, shape (nrh, nz)
            Global average across all processes.
        field_std : ndarray, shape (nrh, nz) — only if return_std is True
        """
        f = field.ravel()
        idx = self.rho_indices.ravel() * self.nz + self.z_indices.ravel()

        sum_f_loc = np.bincount(idx, weights=f, minlength=self.nrh * self.nz)
        sum_n_loc = np.bincount(idx, minlength=self.nrh * self.nz)

        if mpi.nb_proc > 1:
            sum_f_all = mpi.comm.gather(sum_f_loc, root=0)
            sum_n_all = mpi.comm.gather(sum_n_loc, root=0)

            if mpi.rank == 0:
                sum_f = np.sum(sum_f_all, axis=0)
                sum_n = np.sum(sum_n_all, axis=0)
            else:
                sum_f = None
                sum_n = None

            sum_f = mpi.comm.bcast(sum_f, root=0)
            sum_n = mpi.comm.bcast(sum_n, root=0)
        else:
            sum_f = sum_f_loc
            sum_n = sum_n_loc

        mask_nonzero = sum_n > 0

        avg_flat = np.zeros(self.nrh * self.nz)
        avg_flat[mask_nonzero] = sum_f[mask_nonzero] / sum_n[mask_nonzero]

        field_avg = avg_flat.reshape(self.nrh, self.nz)

        if not return_std:
            return field_avg

        sum_f2_loc = np.bincount(idx, weights=f**2, minlength=self.nrh * self.nz)

        if mpi.nb_proc > 1:
            sum_f2_all = mpi.comm.gather(sum_f2_loc, root=0)

            if mpi.rank == 0:
                sum_f2 = np.sum(sum_f2_all, axis=0)
            else:
                sum_f2 = None

            sum_f2 = mpi.comm.bcast(sum_f2, root=0)
        else:
            sum_f2 = sum_f2_loc

        f2_flat = np.zeros(self.nrh * self.nz)
        f2_flat[mask_nonzero] = sum_f2[mask_nonzero] / sum_n[mask_nonzero]
        var_flat = np.maximum(f2_flat - avg_flat**2, 0.0)
        std_flat = np.sqrt(var_flat)
        field_std = std_flat.reshape(self.nrh, self.nz)

        return field_avg, field_std

    def compute_volume_weights(self):
        """Compute the volume weight of each mesh cell

        For a uniform isotropic mesh (dx=dy=dz=d), all cells have the same
        volume d^3. The grid spacing is read from the operator if available.

        Returns
        -------
        weights : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Volume weights on this process.
        """
        d = self.oper.delta if hasattr(self.oper, "delta") else 1.0
        return np.full_like(self.X, d**3)
