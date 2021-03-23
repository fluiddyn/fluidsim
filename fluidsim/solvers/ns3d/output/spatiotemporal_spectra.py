"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS3D
   :members:
   :private-members:

"""

from pathlib import Path

from math import pi
import numpy as np
from scipy import signal
import h5py

from fluidsim.base.output.spatiotemporal_spectra import SpatioTemporalSpectra


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectra):
    def loop_spectra_kzkhomega(self, spectrum_k0k1k2omega, khs, KH, kzs, KZ):
        """Compute the kz-kh-omega spectrum."""
        if spectrum_k0k1k2omega.dtype == "complex64":
            dtype = "float32"
        else:
            dtype = "float64"
        deltakh = khs[1]
        deltakz = kzs[1]
        nkh = len(khs)
        nkz = len(kzs)
        nk0, nk1, nk2, nomega = spectrum_k0k1k2omega.shape
        spectrum_kzkhomega = np.zeros((nkz, nkh, nomega), dtype=dtype)
        for ik0 in range(nk0):
            for ik1 in range(nk1):
                for ik2 in range(nk2):
                    value = spectrum_k0k1k2omega[ik0, ik1, ik2, :]
                    kappa = KH[ik0, ik1, ik2]
                    ikh = int(kappa / deltakh)
                    kz = abs(KZ[ik0, ik1, ik2])
                    ikz = int(round(kz / deltakz))
                    if ikz >= nkz - 1:
                        ikz = nkz - 1
                    if ikh >= nkh - 1:
                        ikh = nkh - 1
                        spectrum_kzkhomega[ikz, ikh, :] += value
                    else:
                        coef_share = (kappa - khs[ikh]) / deltakh
                        spectrum_kzkhomega[ikz, ikh, :] += (
                            1 - coef_share
                        ) * value
                        spectrum_kzkhomega[ikz, ikh + 1, :] += coef_share * value

        # get one-sided spectrum in the omega dimension
        nomega = nomega // 2 + 1
        spectrum_onesided = np.zeros((nkz, nkh, nomega))
        spectrum_onesided[:, :, 0] = spectrum_kzkhomega[:, :, 0]
        spectrum_onesided[:, :, 1:] = 0.5 * (
            spectrum_kzkhomega[:, :, 1:nomega]
            + spectrum_kzkhomega[:, :, -1:-nomega:-1]
        )
        return spectrum_onesided / (deltakz * deltakh)

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
        oper = self.sim.oper
        order = spectra["dims_order"]
        KZ = oper.deltakz * spectra[f"K{order[0]}_adim"]
        KY = oper.deltaky * spectra[f"K{order[1]}_adim"]
        KX = oper.deltakx * spectra[f"K{order[2]}_adim"]
        KH = np.sqrt(KX ** 2 + KY ** 2)

        kz_spectra = np.arange(0, KZ.max() + 1e-15, oper.deltakz)

        deltakh = oper.deltakh
        khmax_spectra = min(KX.max(), KY.max())
        nkh_spectra = max(2, int(khmax_spectra / deltakh))
        kh_spectra = deltakh * np.arange(nkh_spectra)

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
            if not key.startswith("spect"):
                continue
            spectra_kzkhomega[key] = self.loop_spectra_kzkhomega(
                data, kh_spectra, KH, kz_spectra, KZ
            )

        # total kinetic energy
        spectra_kzkhomega["spect_K"] = 0.5 * (
            spectra_kzkhomega["spect_vx"]
            + spectra_kzkhomega["spect_vy"]
            + spectra_kzkhomega["spect_vz"]
        )

        # potential energy
        try:
            N = self.sim.params.N
            spectra_kzkhomega["spect_A"] = (
                0.5 / N ** 2 * spectra_kzkhomega["spect_b"]
            )
        except AttributeError:
            pass

        # save to file
        path_file = Path(self.sim.output.path_run) / "spatiotemporal_spectra.h5"
        with h5py.File(path_file, "w") as file:
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in spectra_kzkhomega.items():
                file.create_dataset(key, data=val)

        # toroidal/poloidal decomposition
        if save_urud:
            print("Computing ur, ud spectra...")
            spectra = self.compute_spectra_urud(tmin=tmin, tmax=tmax, dtype=dtype)
            spectra_kzkhomega = {}

            for key, data in spectra.items():
                if not key.startswith("spect"):
                    continue
                spectra_kzkhomega[key] = self.loop_spectra_kzkhomega(
                    data, kh_spectra, KH, kz_spectra, KZ
                )

            with h5py.File(path_file, "a") as file:
                for key, val in spectra_kzkhomega.items():
                    file.create_dataset(key, data=val)

    def plot_kzkhomega(
        self,
        key_field=None,
        tmin=0,
        tmax=None,
        dtype=None,
        equation=None,
        cmap=None,
        vmin=None,
        vmax=None,
    ):
        """plot the spatiotemporal spectra, with a cylindrical average in k-space"""
        if key_field is None:
            key_field = self.keys_fields[0]
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end
        if cmap is None:
            cmap = "viridis"

        path_file = path_file = (
            Path(self.sim.output.path_run) / "spatiotemporal_spectra.h5"
        )

        spectra_kzkhomega = {}

        key_spect = "spect_" + key_field

        # load spectra from file if it exists
        if path_file.is_file():
            # we should check if times match?
            print("loading spectra from file...")
            with h5py.File(path_file, "r") as file:
                if dtype == "complex64":
                    spectra_kzkhomega[key_spect] = file[key_spect][...].astype(
                        "float32"
                    )
                elif dtype == "complex128":
                    spectra_kzkhomega[key_spect] = file[key_spect][...].astype(
                        "float64"
                    )
                else:
                    spectra_kzkhomega[key_spect] = file[key_spect][...]
        else:
            # compute spectra and save to file, then load
            if key_spect.startswith("spect_Kh"):
                save_urud = True
            else:
                save_urud = False
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
            xaxis, yaxis, np.log10(spect), cmap=cmap, vmin=vmin, vmax=vmax
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
        series = self.load_time_series(tmin=tmin, tmax=tmax, dtype=dtype)

        # get the sampling frequency
        times = series["times"]
        f_sample = 1 / np.mean(times[1:] - times[:-1])

        # toroidal/poloidal decomposition
        # urx_fft, ury_fft contain shear modes!
        vx_fft = series["spect_vx"]
        vy_fft = series["spect_vy"]
        if vx_fft.dtype == "complex64":
            dtype = "float32"
        else:
            dtype = "float64"

        oper = self.sim.oper
        order = series["dims_order"]
        KY = (oper.deltaky * series[f"K{order[1]}_adim"]).astype(dtype)
        KX = (oper.deltakx * series[f"K{order[2]}_adim"]).astype(dtype)

        udx_fft = np.zeros_like(vx_fft)
        udy_fft = np.zeros_like(vx_fft)
        urx_fft = np.zeros_like(vx_fft)
        ury_fft = np.zeros_like(vx_fft)

        inv_Kh_square_nozero = KX ** 2 + KY ** 2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        for it in range(times.size):
            kdotu_fft = KX * vx_fft[:, :, :, it] + KY * vy_fft[:, :, :, it]
            udx_fft[:, :, :, it] = kdotu_fft * KX * inv_Kh_square_nozero
            udy_fft[:, :, :, it] = kdotu_fft * KY * inv_Kh_square_nozero

            urx_fft[:, :, :, it] = vx_fft[:, :, :, it] - udx_fft[:, :, :, it]
            ury_fft[:, :, :, it] = vy_fft[:, :, :, it] - udy_fft[:, :, :, it]

        # perform time fft
        print("computing temporal spectra...")

        spectra = {k: v for k, v in series.items() if k.startswith("K")}

        # ud
        spectra["spect_Khd"] = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = signal.periodogram(udx_fft, fs=f_sample)
        spectra["spect_Khd"] += 0.5 * spectrum
        freq, spectrum = signal.periodogram(udy_fft, fs=f_sample)
        spectra["spect_Khd"] += 0.5 * spectrum

        # ur
        spectra["spect_Khr"] = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = signal.periodogram(urx_fft, fs=f_sample)
        spectra["spect_Khr"] += 0.5 * spectrum
        freq, spectrum = signal.periodogram(ury_fft, fs=f_sample)
        spectra["spect_Khr"] += 0.5 * spectrum

        spectra["omegas"] = 2 * pi * freq
        spectra["dims_order"] = order

        return spectra
