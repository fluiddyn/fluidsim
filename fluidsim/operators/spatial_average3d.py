"""Spatial radial and azimuthal average in 3D (:mod:`fluidsim.operator.spatial_average3d`)
==========================================================================================

Provides:

.. autoclass:: SpatialAverage
   :members:
   :private-members:

"""

import numpy as np
from fluiddyn.util import mpi

from fluidfft.fft3d.operators import loop_spectra3d

from fluidsim.operators.coord_system3d import CoordSystem3DConverter


def loop_spectra_kzkh(spectrum_k0k1k2, khs, KH, kzs, KZ):
    """Compute the kz-kh spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1] - kzs[0]
    kz_min = kzs[0]
    nkh = len(khs)
    nkz = len(kzs)
    spectrum_kzkh = np.zeros((nkz, nkh))
    nk0, nk1, nk2 = spectrum_k0k1k2.shape
    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                value = spectrum_k0k1k2[ik0, ik1, ik2]
                kappa = KH[ik0, ik1, ik2]
                ikh = int(kappa / deltakh)
                kz = KZ[ik0, ik1, ik2]
                ikz = int(round((kz - kz_min) / deltakz))
                if ikz >= nkz - 1:
                    ikz = nkz - 1
                if ikh >= nkh - 1:
                    ikh = nkh - 1
                    spectrum_kzkh[ikz, ikh] += value
                else:
                    coef_share = (kappa - khs[ikh]) / deltakh
                    spectrum_kzkh[ikz, ikh] += (1 - coef_share) * value
                    spectrum_kzkh[ikz, ikh + 1] += coef_share * value
    return spectrum_kzkh


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
        """Compute cylindrical and spherical coordinate arrays

        Uses CoordSystem3DConverter to compute rho and r.
        """
        self.rho = self.coord_conv.rh
        self.r = self.coord_conv.r_not0

    def _prepare_radial_bins(self):
        """Prepare bins for radial averaging

        Creates uniformly spaced bin at deltar = dr * deltax centers spanning [r_min, r_max] globally.
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
        """Prepare bins for azimuthal averaging

        Creates uniformly spaced bin centers for rho and z.
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
        """Compute the total weight (count) in each bin across all processes.

        This computes the normalization factor for averaging.
        For radial average: counts points in each spherical shell.
        For azimuthal average: counts points in each (rho, z) bin.
        """
        ones_field = np.ones_like(self.X)

        radial_weights_loc = loop_spectra3d(ones_field, self.r_centers, self.r**2)

        azimuthal_weights_loc = loop_spectra_kzkh(
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

        Implements the spherical average:

            <f>_Omega(r) = 1/(4*pi) * integral f(r,theta,phi) sin(phi) dtheta dphi

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
        """Average over radial bins for a scalar field using loop_spectra3d.

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
        sum_f_loc = loop_spectra3d(field, self.r_centers, self.r**2)

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
        sum_f2_loc = loop_spectra3d(field**2, self.r_centers, self.r**2)

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

        Implements the azimuthal average over the angle theta:

            <f>_theta(rho, z) = 1/(2*pi) * integral f(rho, theta, z) dtheta

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
        """Average over (rho, z) bins for a scalar field using loop_spectra_kzkh.

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
        sum_f_loc = loop_spectra_kzkh(
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
        sum_f2_loc = loop_spectra_kzkh(
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
        """Compute the volume weight of each mesh cell

        For a uniform isotropic mesh (dx=dy=dz=d), all cells have the same
        volume d^3. The grid spacing is read from the operator if available.

        Returns
        -------
        Volumes : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Volume weights on this process.
        Volumes : ndarray, shape (Nx_loc, Ny_loc, Nz_loc)
            Volume weights on this process.
        """
        d = self.oper.delta if hasattr(self.oper, "delta") else 1.0
        return np.full_like(self.X, d**3)
