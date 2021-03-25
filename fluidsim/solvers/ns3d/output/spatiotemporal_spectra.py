"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS3D
   :members:
   :private-members:

"""

from math import pi

import numpy as np
from scipy import signal
import h5py

from fluidsim.base.output.spatiotemporal_spectra import SpatioTemporalSpectra

from transonic import boost, Array, Type

A4 = Array[Type(np.float32, np.float64), "4d", "C"]
A3 = "float[:,:,:]"
A1 = "float[:]"


@boost
def loop_spectra_kzkhomega(
    spectrum_k0k1k2omega: A4, khs: A1, KH: A3, kzs: A1, KZ: A3
):
    """Compute the kz-kh-omega spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1]

    nkh = len(khs)
    nkz = len(kzs)
    nk0, nk1, nk2, nomega = spectrum_k0k1k2omega.shape
    spectrum_kzkhomega = np.zeros(
        (nkz, nkh, nomega), dtype=spectrum_k0k1k2omega.dtype
    )

    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                values = spectrum_k0k1k2omega[ik0, ik1, ik2, :]
                kappa = KH[ik0, ik1, ik2]
                ikh = int(kappa / deltakh)
                kz = abs(KZ[ik0, ik1, ik2])
                ikz = int(round(kz / deltakz))
                if ikz >= nkz - 1:
                    ikz = nkz - 1
                if ikh >= nkh - 1:
                    ikh = nkh - 1
                    for i, value in enumerate(values):
                        spectrum_kzkhomega[ikz, ikh, i] += value
                else:
                    coef_share = (kappa - khs[ikh]) / deltakh
                    for i, value in enumerate(values):
                        spectrum_kzkhomega[ikz, ikh, i] += (
                            1 - coef_share
                        ) * value
                        spectrum_kzkhomega[ikz, ikh + 1, i] += coef_share * value

    # get one-sided spectrum in the omega dimension
    nomega = nomega // 2 + 1
    spectrum_onesided = np.zeros((nkz, nkh, nomega))
    spectrum_onesided[:, :, 0] = spectrum_kzkhomega[:, :, 0]
    spectrum_onesided[:, :, 1:] = 0.5 * (
        spectrum_kzkhomega[:, :, 1:nomega]
        + spectrum_kzkhomega[:, :, -1:-nomega:-1]
    )
    return spectrum_onesided / (deltakz * deltakh)


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectra):
    def _get_path_saved_spectra(self, tmin, tmax, dtype, save_urud):
        base = f"spatiotemporal_spectra_{tmin}_{tmax}"
        if dtype is not None:
            base += "_{dtype}"
        if save_urud:
            base += "_urud"
        return self.path_dir / (base + ".h5")

    def save_spectra_kzkhomega(
        self, tmin=0, tmax=None, dtype=None, save_urud=False
    ):
        """save the spatiotemporal spectra, with a cylindrical average in k-space"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # compute spectra
        print("Computing spectra...")
        spectra = self.compute_spectra(tmin=tmin, tmax=tmax, dtype=dtype)

        # get kz, kh
        params_oper = self.sim.params.oper
        deltakz = 2 * pi / params_oper.Lz
        deltaky = 2 * pi / params_oper.Ly
        deltakx = 2 * pi / params_oper.Lx
        order = spectra["dims_order"]
        KZ = deltakz * spectra[f"K{order[0]}_adim"]
        KY = deltaky * spectra[f"K{order[1]}_adim"]
        KX = deltakx * spectra[f"K{order[2]}_adim"]
        KH = np.sqrt(KX ** 2 + KY ** 2)

        kz_spectra = np.arange(0, KZ.max() + 1e-15, deltakz)

        deltakh = max(deltakx, deltaky)
        khmax_spectra = min(KX.max(), KY.max())
        nkh_spectra = max(2, int(khmax_spectra / deltakh))
        kh_spectra = deltakh * np.arange(nkh_spectra)

        del KX, KY

        # get one-sided frequencies
        omegas = spectra["omegas"]
        nomegas = omegas.size // 2 + 1
        omegas_onesided = abs(omegas[:nomegas])

        # perform cylindrical average
        spectra_kzkhomega = {
            "kz_spectra": kz_spectra,
            "kh_spectra": kh_spectra,
            "omegas": omegas_onesided,
        }
        for key, data in spectra.items():
            if not key.startswith("spectrum_"):
                continue
            spectra_kzkhomega[key] = loop_spectra_kzkhomega(
                np.ascontiguousarray(data), kh_spectra, KH, kz_spectra, KZ
            )

        del spectra

        # total kinetic energy
        spectra_kzkhomega["spectrum_K"] = 0.5 * (
            spectra_kzkhomega["spectrum_vx"]
            + spectra_kzkhomega["spectrum_vy"]
            + spectra_kzkhomega["spectrum_vz"]
        )

        # potential energy
        try:
            N = self.sim.params.N
            spectra_kzkhomega["spectrum_A"] = (
                0.5 / N ** 2 * spectra_kzkhomega["spectrum_b"]
            )
        except AttributeError:
            pass

        # save to file
        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        with h5py.File(path_file, "w") as file:
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in spectra_kzkhomega.items():
                file.create_dataset(key, data=val)

        # toroidal/poloidal decomposition
        if save_urud:
            print("Computing ur, ud spectra...")
            spectra_urud_kzkhomega = {}
            spectra = self.compute_spectra_urud(tmin=tmin, tmax=tmax, dtype=dtype)

            for key, data in spectra.items():
                if not key.startswith("spectrum_"):
                    continue
                spectra_urud_kzkhomega[key] = loop_spectra_kzkhomega(
                    np.ascontiguousarray(data), kh_spectra, KH, kz_spectra, KZ
                )
                spectra_kzkhomega[key] = spectra_urud_kzkhomega[key]

            with h5py.File(path_file, "a") as file:
                for key, val in spectra_urud_kzkhomega.items():
                    file.create_dataset(key, data=val)

        return spectra_kzkhomega

    def plot_kzkhomega(
        self,
        key_field=None,
        tmin=0,
        tmax=None,
        dtype=None,
        equation=None,
        cmap="viridis",
        vmin=None,
        vmax=None,
    ):
        """plot the spatiotemporal spectra, with a cylindrical average in k-space"""
        if key_field is None:
            key_field = self.keys_fields[0]
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        key_spect = "spectrum_" + key_field
        if key_spect.startswith("spectrum_Kh"):
            save_urud = True
        else:
            save_urud = False

        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)

        spectra_kzkhomega = {}

        if path_file.exists():
            # we should check if times match?
            print("loading spectra from file...")
            with h5py.File(path_file, "r") as file:
                spectrum = file[key_spect][...]
                if dtype == "complex64":
                    float_dtype = "float32"
                elif dtype == "complex128":
                    float_dtype = "float64"
                if dtype:
                    spectrum = spectrum.astype(float_dtype)
                spectra_kzkhomega[key_spect] = spectrum
                spectra_kzkhomega["kh_spectra"] = file["kh_spectra"][...]
                spectra_kzkhomega["kz_spectra"] = file["kz_spectra"][...]
                spectra_kzkhomega["omegas"] = file["omegas"][...]
        else:
            # compute spectra and save to file, then load
            self.save_spectra_kzkhomega(
                tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
            )
            with h5py.File(path_file, "r") as file:
                for key in file.keys():
                    spectra_kzkhomega[key] = file[key][...]

        # slice along equation
        if equation is None:
            equation = f"omega=0"
        if equation.startswith("omega="):
            omega = eval(equation[len("omega=") :])
            omegas = spectra_kzkhomega["omegas"]
            iomega = abs(omegas - omega).argmin()
            spect = spectra_kzkhomega[key_spect][:, :, iomega]
            xaxis = spectra_kzkhomega["kh_spectra"]
            yaxis = spectra_kzkhomega["kz_spectra"]
            xlabel = r"$k_h$"
            ylabel = r"$k_z$"
            omega = omegas[iomega]
            equation = r"$\omega=$" + f"{omega:.2g}"
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                equation = r"$\omega/N=$" + f"{omega/N:.2g}"
            except AttributeError:
                pass
        elif equation.startswith("kh="):
            kh = eval(equation[len("kh=") :])
            kh_spectra = spectra_kzkhomega["kh_spectra"]
            ikh = abs(kh_spectra - kh).argmin()
            spect = spectra_kzkhomega[key_spect][:, ikh, :].transpose()

            xaxis = spectra_kzkhomega["kz_spectra"]
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_z$"
            ylabel = r"$\omega/N$"
            kh = kh_spectra[ikh]
            equation = r"$k_h=$" + f"{kh:.2g}"
        elif equation.startswith("kz="):
            kz = eval(equation[len("kz=") :])
            kz_spectra = spectra_kzkhomega["kz_spectra"]
            ikz = abs(kz_spectra - kz).argmin()
            spect = spectra_kzkhomega[key_spect][ikz, :, :].transpose()

            xaxis = spectra_kzkhomega["kh_spectra"]
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_h$"
            ylabel = r"$\omega/N$"
            kz = kz_spectra[ikz]
            equation = r"$k_z=$" + f"{kz:.2g}"
        else:
            raise NotImplementedError(
                "equation must start with 'omega=', 'kh=' or 'kz='"
            )

        # plot
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if vmin is None:
            vmin = np.log10(spect[np.isfinite(spect)].min())
        if vmax is None:
            vmax = np.log10(spect[np.isfinite(spect)].max())

        im = ax.pcolormesh(
            xaxis,
            yaxis,
            np.log10(spect),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
        fig.colorbar(im)

        ax.set_title(
            f"{key_field} spatiotemporal spectra {equation}\n"
            f"tmin={tmin:.2g}, tmax={tmax:.2g}\n" + self.output.summary_simul
        )

        # add dispersion relation : omega = N * kh / sqrt(kh ** 2 + kz ** 2)
        try:
            N = self.sim.params.N
        except AttributeError:
            return
        if equation.startswith(r"$\omega"):
            kz_disp = (N ** 2 / omega ** 2 - 1) * xaxis
            ax.step(xaxis, kz_disp, "k", linewidth=2)
        elif equation.startswith(r"$k_h"):
            omega_disp = kh / np.sqrt(kh ** 2 + xaxis ** 2)
            ax.step(xaxis, omega_disp, "k", linewidth=2)
        elif equation.startswith(r"$k_z"):
            omega_disp = xaxis / np.sqrt(xaxis ** 2 + kz ** 2)
            ax.step(xaxis, omega_disp, "k", linewidth=2)
        else:
            raise ValueError("wrong equation for dispersion relation")

        # reset axis limits after plotting dispersion relation
        ax.set_xlim((xaxis.min(), xaxis.max()))
        ax.set_ylim((yaxis.min(), yaxis.max()))

    def compute_spectra_urud(self, tmin=0, tmax=None, dtype=None):
        """compute the spectra of ur, ud from files"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # load time series as state_spect arrays + times
        series = self.load_time_series(
            keys=("vx", "vy"), tmin=tmin, tmax=tmax, dtype=dtype
        )

        # get the sampling frequency
        times = series["times"]
        f_sample = 1 / np.mean(times[1:] - times[:-1])

        # toroidal/poloidal decomposition
        # urx_fft, ury_fft contain shear modes!
        vx_fft = series["vx_Fourier"]
        vy_fft = series["vy_Fourier"]
        if vx_fft.dtype == "complex64":
            float_dtype = "float32"
        elif vx_fft.dtype == "complex128":
            float_dtype = "float64"

        params_oper = self.sim.params.oper
        deltaky = 2 * pi / params_oper.Ly
        deltakx = 2 * pi / params_oper.Lx

        order = series["dims_order"]

        shapeK = series[f"K{order[1]}_adim"].shape
        KY = np.zeros(shapeK + (1,), dtype=float_dtype)
        KX = np.zeros(shapeK + (1,), dtype=float_dtype)
        KY[..., 0] = deltaky * series[f"K{order[1]}_adim"]
        KX[..., 0] = deltakx * series[f"K{order[2]}_adim"]

        inv_Kh_square_nozero = KX ** 2 + KY ** 2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        kdotu_fft = KX * vx_fft + KY * vy_fft
        udx_fft = kdotu_fft * KX * inv_Kh_square_nozero
        udy_fft = kdotu_fft * KY * inv_Kh_square_nozero

        urx_fft = vx_fft - udx_fft
        ury_fft = vy_fft - udy_fft

        del vx_fft, vy_fft, KX, KY, inv_Kh_square_nozero, kdotu_fft

        # perform time fft
        print("computing temporal spectra...")

        spectra = {k: v for k, v in series.items() if k.startswith("K")}
        del series

        # ud
        spectra["spectrum_Khd"] = Khd = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = self._compute_spectrum(udx_fft)
        Khd += 0.5 * spectrum
        freq, spectrum = self._compute_spectrum(udy_fft)
        Khd += 0.5 * spectrum

        # ur
        spectra["spectrum_Khr"] = Khr = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = self._compute_spectrum(urx_fft)
        Khr += 0.5 * spectrum
        freq, spectrum = self._compute_spectrum(ury_fft)
        Khr += 0.5 * spectrum

        spectra["omegas"] = 2 * pi * freq
        spectra["dims_order"] = order

        return spectra
