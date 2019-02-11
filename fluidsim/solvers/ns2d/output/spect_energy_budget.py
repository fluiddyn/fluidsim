"""Energy budget (:mod:`fluidsim.solvers.ns2d.output.spect_energy_budget`)
==========================================================================

.. autoclass:: SpectralEnergyBudgetNS2D
   :members:
   :private-members:

"""

import numpy as np
import h5py


from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase,
    cumsum_inv,
)


class SpectralEnergyBudgetNS2D(SpectralEnergyBudgetBase):
    """Save and plot energy budget in spectral space."""

    def compute(self):
        """compute the spectral energy budget at one time."""
        oper = self.sim.oper

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = oper.ifft2(px_rot_fft)
        py_rot = oper.ifft2(py_rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Frot = -ux * px_rot - uy * (py_rot + self.params.beta)
        Frot_fft = oper.fft2(Frot)
        oper.dealiasing(Frot_fft)

        Fx = -ux * px_ux - uy * (py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux * px_uy - uy * (py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        transferZ_fft = (
            np.real(rot_fft.conj() * Frot_fft + rot_fft * Frot_fft.conj()) / 2.0
        )
        # print ('sum(transferZ) = {0:9.4e} ; sum(abs(transferZ)) = {1:9.4e}'
        #       ).format(self.sum_wavenumbers(transferZ_fft),
        #                self.sum_wavenumbers(abs(transferZ_fft)))

        transferE_fft = (
            np.real(
                ux_fft.conj() * Fx_fft
                + ux_fft * Fx_fft.conj()
                + uy_fft.conj() * Fy_fft
                + uy_fft * Fy_fft.conj()
            )
            / 2.0
        )
        # print ('sum(transferE) = {0:9.4e} ; sum(abs(transferE)) = {1:9.4e}'
        #       ).format(self.sum_wavenumbers(transferE_fft),
        #                self.sum_wavenumbers(abs(transferE_fft)))

        transfer2D_E = self.spectrum2D_from_fft(transferE_fft)
        transfer2D_Z = self.spectrum2D_from_fft(transferZ_fft)

        dict_results = {
            "transfer2D_E": transfer2D_E,
            "transfer2D_Z": transfer2D_Z,
        }
        return dict_results

    def _online_plot_saving(self, dict_results):
        transfer2D_E = dict_results["transfer2D_E"]
        transfer2D_Z = dict_results["transfer2D_Z"]
        khE = self.oper.khE
        PiE = cumsum_inv(transfer2D_E) * self.oper.deltak
        PiZ = cumsum_inv(transfer2D_Z) * self.oper.deltak
        self.axe_a.plot(khE + khE[1], PiE, "k")
        self.axe_b.plot(khE + khE[1], PiZ, "g")

    def plot(self, tmin=0, tmax=1000, delta_t=2):

        with h5py.File(self.path_file, "r") as h5file:
            dset_times = h5file["times"]
            dset_khE = h5file["khE"]
            khE = dset_khE[...]
            khE = khE + khE[1]

            dset_transferE = h5file["transfer2D_E"]
            dset_transferZ = h5file["transfer2D_Z"]

            # nb_spectra = dset_times.shape[0]
            times = dset_times[...]
            # nt = len(times)

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))

            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            to_print = "plot(tmin={}, tmax={}, delta_t={:.2f})".format(
                tmin, tmax, delta_t
            )
            print(to_print)

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]
            print(
                """plot spectral energy budget
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                    tmin_plot,
                    tmax_plot,
                    delta_t,
                    imin_plot,
                    imax_plot,
                    delta_i_plot,
                )
            )

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel("spectra")
            ax1.set_xscale("log")
            ax1.set_yscale("linear")

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot, delta_i_plot):

                    transferE = dset_transferE[it]
                    transferZ = dset_transferZ[it]

                    PiE = cumsum_inv(transferE) * self.oper.deltak
                    PiZ = cumsum_inv(transferZ) * self.oper.deltak

                    ax1.plot(khE, PiE, "k", linewidth=1)
                    ax1.plot(khE, PiZ, "g", linewidth=1)

            transferE = dset_transferE[imin_plot:imax_plot].mean(0)
            transferZ = dset_transferZ[imin_plot:imax_plot].mean(0)

        PiE = cumsum_inv(transferE) * self.oper.deltak
        PiZ = cumsum_inv(transferZ) * self.oper.deltak

        ax1.plot(khE, PiE, "r", linewidth=2)
        ax1.plot(khE, PiZ, "m", linewidth=2)
