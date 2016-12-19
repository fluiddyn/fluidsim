"""Energy budget (:mod:`fluidsim.solvers.ns2d.strat.output.spect_energy_budget)
==========================================================================

.. autoclass:: SpectralEnergyBudgetNS2DStrat
   :members:
   :private-members:

"""

import numpy as np
import h5py


from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase, cumsum_inv)

class SpectralEnergyBudgetNS2DStrat(SpectralEnergyBudgetBase):
    """Save and plot energy budget in spectral space."""

    def compute(self):
        """compute the spectral energy budget at one time."""
        oper = self.sim.oper

        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')

        rot_fft = self.sim.state.state_fft.get_var('rot_fft')
        b_fft = self.sim.state.state_fft.get_var('b_fft')
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_b_fft, py_b_fft = oper.gradfft_from_fft(b_fft)
        px_b = oper.ifft2(px_b_fft)
        py_b = oper.ifft2(py_b_fft)

        Fb = -ux*px_b - uy*py_b
        Fb_fft = oper.fft2(Fb)
        oper.dealiasing(Fb_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = oper.ifft2(px_rot_fft)
        py_rot = oper.ifft2(py_rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Frot = -ux*px_rot - uy*(py_rot + self.params.beta)
        Frot_fft = oper.fft2(Frot)
        oper.dealiasing(Frot_fft)

        Fx = -ux*px_ux - uy*(py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux*px_uy - uy*(py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        transferZ_fft = np.real(rot_fft.conj()*Frot_fft +
                                rot_fft*Frot_fft.conj())/2.
        # print ('sum(transferZ) = {0:9.4e} ; sum(abs(transferZ)) = {1:9.4e}'
        #       ).format(self.sum_wavenumbers(transferZ_fft),
        #                self.sum_wavenumbers(abs(transferZ_fft)))

        transferEK_fft = np.real(ux_fft.conj()*Fx_fft + uy_fft.conj()*Fy_fft)

        B_fft = np.real(uy_fft.conj()*b_fft)

        transferEA_fft = (1/self.params.N**2) * np.real(b_fft.conj()*Fb_fft)

        # Square of the modulus of the velocity (extract first wavenumber k=0)
        square_modulus_vel = ux_fft.conj()*ux_fft + uy_fft.conj()*uy_fft
        square_modulus_vel = square_modulus_vel[:, 1:]
        square_modulus_b = b_fft.conj()*b_fft
        square_modulus_b = square_modulus_b[:, 1:]

        disipationEK = self.params.nu_2 * (
            self.output.oper.kxE**2 + self.output.oper.kyE**2) * (
                square_modulus_vel)

        disipationEA = (self.params.nu_2/self.params.N**2) * (
             self.output.oper.kxE**2 + self.output.oper.kyE**2) * (
                 square_modulus_b)

        # print ('sum(transferEK) = {0:9.4e} ; sum(abs(transferEK)) = {1:9.4e}'
        # ).format(self.sum_wavenumbers(transferEK_fft),
        # self.sum_wavenumbers(abs(transferEK_fft)))

        # Transfer spectrum 1D Kinetic energy
        transferEK_kx, transferEK_ky = self.spectra1D_from_fft(transferEK_fft)
        transferEA_kx, transferEA_ky = self.spectra1D_from_fft(transferEA_fft)
        B_kx, B_ky = self.spectra1D_from_fft(B_fft)
        disipationEK_kx, disipationEK_ky = self.spectra1D_from_fft(
             disipationEK)
        disipationEA_kx, disipationEA_ky = self.spectra1D_from_fft(
            disipationEA)

        # Transfer spectrum shell mean
        transferEK_2d = self.spectrum2D_from_fft(transferEK_fft)
        transferEA_2d = self.spectrum2D_from_fft(transferEA_fft)
        B_2d = self.spectrum2D_from_fft(B_fft)
        # disipationEK_2d = self.spectrum2D_from_fft(disipationEK)
        # disipationEA_2d = self.spectrum2D_from_fft(disipationEA)

        transferZ_2d = self.spectrum2D_from_fft(transferZ_fft)

        dico_results = {
            'transferEK_kx': transferEK_kx,
            'transferEK_ky': transferEK_ky,
            'transferEK_2d': transferEK_2d,
            'transferEA_kx': transferEA_kx,
            'transferEA_ky': transferEA_ky,
            'transferEA_2d': transferEA_2d,
            'transferZ_2d': transferZ_2d,
            'B_kx': B_kx,
            'B_ky': B_ky,
            'B_2d': B_2d,
            'disipationEK_kx': disipationEK_kx,
            'disipationEK_ky': disipationEK_ky,
            'disipationEA_kx': disipationEA_kx,
            'disipationEA_ky': disipationEA_ky}
            # 'disipationEK_2d': disipationEK_2d}

        small_value = 1e-16
        for k, v in dico_results.items():
            if k.startswith('transfer'):
                assert abs(v.sum()) < small_value

        return dico_results

    def _online_plot(self, dico_results):
        transfer2D_E = dico_results['transfer2D_E']
        transfer2D_Z = dico_results['transfer2D_Z']
        khE = self.oper.khE
        PiE = cumsum_inv(transfer2D_E)*self.oper.deltakh
        PiZ = cumsum_inv(transfer2D_Z)*self.oper.deltakh
        self.axe_a.plot(khE+khE[1], PiE, 'k')
        self.axe_b.plot(khE+khE[1], PiZ, 'g')

    def plot(self, tmin=0, tmax=1000, delta_t=2):

        f = h5py.File(self.path_file, 'r')
        dset_times = f['times']
        dset_kxE = f['kxE']
        kxE = dset_kxE.value
        kxE = kxE+kxE[1]

        # dset_transferE = f['transfer2D_E']
        # dset_transferZ = f['transfer2D_Z']
        dset_transferEK_kx = f['transferEK_kx']


        # nb_spectra = dset_times.shape[0]
        times = dset_times.value
        # nt = len(times)

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))

        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot = 1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        to_print = 'plot(tmin={0}, tmax={1}, delta_t={2:.2f})'.format(
            tmin, tmax, delta_t)
        print(to_print)

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]
        print(
            '''plot spectral energy budget
            tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
            imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
                tmin_plot, tmax_plot, delta_t,
                imin_plot, imax_plot, delta_i_plot))

        fig, ax1 = self.output.figure_axe()
        ax1.set_xlabel('$k_x$')
        ax1.set_ylabel('flux EK')
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('linear')

        if delta_t != 0.:
            for it in range(imin_plot, imax_plot, delta_i_plot):

                # transferE = dset_transferE[it]
                # transferZ = dset_transferZ[it]
                transferEK_kx = dset_transferEK_kx[it]

                # PiE = cumsum_inv(transferE)*self.oper.deltakh
                # PiZ = cumsum_inv(transferZ)*self.oper.deltakh
                PiEK_kx = cumsum_inv(transferEK_kx)*self.oper.deltakx

                # ax1.plot(khE, PiE, 'k', linewidth=1)
                # ax1.plot(khE, PiZ, 'g', linewidth=1)
                ax1.plot(kxE, PiEK_kx, 'g', linewidth=1)

        # transferE = dset_transferE[imin_plot:imax_plot].mean(0)
        # transferZ = dset_transferZ[imin_plot:imax_plot].mean(0)
        transferEK_kx = dset_transferEK_kx[imin_plot:imax_plot].mean(0)

        # PiE = cumsum_inv(transferE)*self.oper.deltakh
        # PiZ = cumsum_inv(transferZ)*self.oper.deltakh
        PiEK_kx = cumsum_inv(transferEK_kx)*self.oper.deltakh

        # ax1.plot(khE, PiE, 'r', linewidth=2)
        # ax1.plot(khE, PiZ, 'm', linewidth=2)
        ax1.plot(kxE, PiEK_kx, 'm--', linewidth=2)
