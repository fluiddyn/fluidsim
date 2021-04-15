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
def compute_spectrum_kzkhomega(
    field_k0k1k2omega: A4, khs: A1, kzs: A1, KX: A3, KZ: A3, KH: A3
):
    """Compute the kz-kh-omega spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1]

    nkh = len(khs)
    nkz = len(kzs)
    nk0, nk1, nk2, nomega = field_k0k1k2omega.shape
    spectrum_kzkhomega = np.zeros(
        (nkz, nkh, nomega), dtype=field_k0k1k2omega.dtype
    )

    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                values = field_k0k1k2omega[ik0, ik1, ik2, :]
                kx = KX[ik0, ik1, ik2]
                if kx != 0.0:
                    # warning: we should also consider another condition
                    # (kx != kx_max) but it is not necessary here mainly
                    # because of dealiasing
                    values = 2 * values

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
    spectrum_onesided[:, :, 1:] = (
        spectrum_kzkhomega[:, :, 1:nomega]
        + spectrum_kzkhomega[:, :, -1:-nomega:-1]
    )
    return spectrum_onesided / (deltakz * deltakh)


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectra):
    def _get_path_saved_spectra(self, tmin, tmax, dtype, save_urud):
        base = f"periodogram_{tmin}_{tmax}"
        if dtype is not None:
            base += f"_{dtype}"
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
        spectra = self.compute_temporal_spectra(tmin=tmin, tmax=tmax, dtype=dtype)

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

        del KY

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
            spectra_kzkhomega[key] = compute_spectrum_kzkhomega(
                np.ascontiguousarray(data), kh_spectra, kz_spectra, KX, KZ, KH
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
                spectra_urud_kzkhomega[key] = compute_spectrum_kzkhomega(
                    np.ascontiguousarray(data), kh_spectra, kz_spectra, KX, KZ, KH
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
        xmax=None,
        ymax=None,
        cmap="viridis",
        vmin=None,
        vmax=None,
    ):
        """plot the spatiotemporal spectra, with a cylindrical average in k-space"""
        keys_plot = self.keys_fields + ["Khd", "Khr", "Kp"]
        if key_field is None:
            key_field = keys_plot[0]
        if key_field not in keys_plot:
            raise KeyError(f"possible keys are {keys_plot}")
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        key_spect = "spectrum_" + key_field
        if key_field.startswith("Kh") or key_field.startswith("Kp"):
            save_urud = True
        else:
            save_urud = False

        path_file = self._get_path_saved_spectra(tmin, tmax, dtype, save_urud)
        path_urud = self._get_path_saved_spectra(tmin, tmax, dtype, True)
        if path_urud.exists() and not path_file.exists():
            path_file = path_urud

        spectra_kzkhomega = {}

        # compute and save spectra if needed
        if not path_file.exists():
            self.save_spectra_kzkhomega(
                tmin=tmin, tmax=tmax, dtype=dtype, save_urud=save_urud
            )

        # load spectrum
        with h5py.File(path_file, "r") as file:
            if key_spect.startswith("spectrum_Kp"):
                spectrum = file["spectrum_Khd"][:] + 0.5 * file["spectrum_vz"][:]
            else:
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

        # slice along equation
        if equation is None:
            equation = f"omega=0"
        if equation.startswith("omega="):
            omega = eval(equation[len("omega=") :])
            omegas = spectra_kzkhomega["omegas"]
            iomega = abs(omegas - omega).argmin()
            spect = spectra_kzkhomega[key_spect][:, :, iomega]
            xaxis = np.arange(spectra_kzkhomega["kh_spectra"].size)
            yaxis = np.arange(spectra_kzkhomega["kz_spectra"].size)
            xlabel = r"$k_h/\delta k_h$"
            ylabel = r"$k_z/\delta k_z$"
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

            xaxis = np.arange(spectra_kzkhomega["kz_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_z/\delta k_z$"
            ylabel = r"$\omega/N$"
            kh = kh_spectra[ikh]
            equation = f"$k_h = {ikh}\\delta k_h = {kh:.2g}$"
        elif equation.startswith("kz="):
            kz = eval(equation[len("kz=") :])
            kz_spectra = spectra_kzkhomega["kz_spectra"]
            ikz = abs(kz_spectra - kz).argmin()
            spect = spectra_kzkhomega[key_spect][ikz, :, :].transpose()

            xaxis = np.arange(spectra_kzkhomega["kh_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_h/\delta k_h$"
            ylabel = r"$\omega/N$"
            kz = kz_spectra[ikz]
            equation = f"$k_z = {ikz}\\delta k_z = {kz:.2g}$"
        elif equation.startswith("ikh="):
            ikh = eval(equation[len("ikh=") :])
            kh_spectra = spectra_kzkhomega["kh_spectra"]
            spect = spectra_kzkhomega[key_spect][:, ikh, :].transpose()

            xaxis = np.arange(spectra_kzkhomega["kz_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_z/\delta k_z$"
            ylabel = r"$\omega/N$"
            kh = kh_spectra[ikh]
            equation = f"$k_h = {ikh}\\delta k_h = {kh:.2g}$"
        elif equation.startswith("ikz="):
            ikz = eval(equation[len("ikz=") :])
            kz_spectra = spectra_kzkhomega["kz_spectra"]
            spect = spectra_kzkhomega[key_spect][ikz, :, :].transpose()

            xaxis = np.arange(spectra_kzkhomega["kh_spectra"].size)
            yaxis = spectra_kzkhomega["omegas"]
            # use reduced frequency for stratified fluids
            try:
                N = self.sim.params.N
                yaxis /= N
            except AttributeError:
                pass

            xlabel = r"$k_h/\delta k_h$"
            ylabel = r"$\omega/N$"
            kz = kz_spectra[ikz]
            equation = f"$k_z = {ikz}\\delta k_z = {kz:.2g}$"
        else:
            raise NotImplementedError(
                "equation must start with 'omega=', 'kh=', 'kz=', 'ikh=' or 'ikz='"
            )

        # plot
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if vmin is None:
            vmin = np.log10(spect[np.isfinite(spect)].min())
        if vmax is None:
            vmax = np.log10(spect[np.isfinite(spect)].max())

        # no log(0)
        spect += 1e-15

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
            f"tmin={tmin:.3f}, tmax={tmax:.3f}\n" + self.output.summary_simul
        )

        # add dispersion relation : omega = N * kh / sqrt(kh ** 2 + kz ** 2)
        try:
            N = self.sim.params.N
        except AttributeError:
            return
        dkz_dkh = (
            spectra_kzkhomega["kz_spectra"][1]
            / spectra_kzkhomega["kh_spectra"][1]
        )
        if equation.startswith(r"$\omega"):
            ikz_disp = np.sqrt(N ** 2 / omega ** 2 - 1) / dkz_dkh * xaxis
            ax.plot(xaxis, ikz_disp, "k+", linewidth=2)
        elif equation.startswith(r"$k_h"):
            omega_disp = ikh / np.sqrt(ikh ** 2 + dkz_dkh ** 2 * xaxis ** 2)
            ax.plot(xaxis, omega_disp, "k+", linewidth=2)
        elif equation.startswith(r"$k_z"):
            omega_disp = xaxis / np.sqrt(xaxis ** 2 + dkz_dkh ** 2 * ikz ** 2)
            ax.plot(xaxis, omega_disp, "k+", linewidth=2)
        else:
            raise ValueError("wrong equation for dispersion relation")

        # set axis limits after plotting dispersion relation
        if xmax is None:
            xmax = xaxis.max()
        if ymax is None:
            ymax = yaxis.max()
        ax.set_xlim((0, xmax))
        ax.set_ylim((0, ymax))

    def compute_spectra_urud(self, tmin=0, tmax=None, dtype=None):
        """compute the spectra of ur, ud from files"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # load time series as state_spect arrays + times
        series = self.load_time_series(
            keys=("vx", "vy"), tmin=tmin, tmax=tmax, dtype=dtype
        )

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
