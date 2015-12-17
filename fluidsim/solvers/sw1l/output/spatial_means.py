
import os
import numpy as np
import matplotlib.pyplot as plt
from fluiddyn.util import mpi
from fluidsim.base.output.spatial_means import (
    SpatialMeansBase, inner_prod)


class SpatialMeansMSW1L(SpatialMeansBase):
    """Handle the saving of spatial mean quantities.

       Viz. total energy, K.E., A.P.E. and Charney potential enstrophy. It also
       handles the computation of forcing and dissipation rates for
       sw1l.modified solver

    """

    def __init__(self, output):

        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super(SpatialMeansMSW1L, self).__init__(output)

    def save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim

        if mpi.rank == 0:
            self.file.write('####\ntime = {0:.6e}\n'.format(tsim))

        energyK_fft, energyA_fft, energyKr_fft = \
            self.output.compute_energies_fft()
        energyK = self.sum_wavenumbers(energyK_fft)
        energyA = self.sum_wavenumbers(energyA_fft)
        energyKr = self.sum_wavenumbers(energyKr_fft)
        energy = energyK + energyA

        CharneyPE_fft = self.output.compute_CharneyPE_fft()
        CharneyPE = self.sum_wavenumbers(CharneyPE_fft)

        if mpi.rank == 0:
            to_print = (
                'E      = {0:11.6e} ; CPE        = {1:11.6e} \n'
                'EK     = {2:11.6e} ; EA         = {3:11.6e} ; '
                'EKr       = {4:11.6e} \n').format(
                    energy, CharneyPE, energyK, energyA, energyKr)
            self.file.write(to_print)

        # Compute and save dissipation rates.
        self.treat_dissipation_rates(energyK_fft, energyA_fft, CharneyPE_fft)

        # Compute and save conversion rates.
        self.treat_conversion()

        # Compute and save skewness and kurtosis.
        eta = self.sim.state.state_phys.get_var('eta')
        meaneta2 = 2./self.c2*energyA
        skew_eta = np.mean(eta**3)/meaneta2**(3./2)
        kurt_eta = np.mean(eta**4)/meaneta2**(2)

        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')
        ux_fft = self.sim.oper.fft2(ux)
        uy_fft = self.sim.oper.fft2(uy)
        rot_fft = self.sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = self.sim.oper.ifft2(rot_fft)
        meanrot2 = self.sum_wavenumbers(abs(rot_fft)**2)
        skew_rot = np.mean(rot**3)/meanrot2**(3./2)
        kurt_rot = np.mean(rot**4)/meanrot2**(2)

        if mpi.rank == 0:
            to_print = (
                'eta skew = {0:11.6e} ; kurt = {1:11.6e} \n'
                'rot skew = {2:11.6e} ; kurt = {3:11.6e} \n').format(
                skew_eta, kurt_eta, skew_rot, kurt_rot)
            self.file.write(to_print)

        if self.sim.params.FORCING:
            self.treat_forcing()

        if mpi.rank == 0:
            self.file.flush()
            os.fsync(self.file.fileno())

        if self.has_to_plot and mpi.rank == 0:
            self.axe_a.plot(tsim, energy, 'k.')
            self.axe_a.plot(tsim, energyK, 'r.')
            self.axe_a.plot(tsim, energyA, 'b.')

            if tsim-self.t_last_show >= self.period_show:
                self.t_last_show = tsim
                fig = self.axe_a.get_figure()
                fig.canvas.draw()

    def treat_conversion(self):
        mean_space = self.sim.oper.mean_space

        c2 = self.sim.params.c2
        eta = self.sim.state('eta')
        div = self.sim.state('div')
        h = eta + 1

        Conv = c2/2*mean_space(h**2*div)
        c2eta1d = c2*mean_space(eta*div)
        c2eta2d = c2*mean_space(eta**2*div)
        c2eta3d = c2*mean_space(eta**3*div)

        if mpi.rank == 0:
            to_print = (
                'Conv = {0:11.6e} ; c2eta1d = {1:11.6e} ; '
                'c2eta2d = {2:11.6e} ; c2eta2d = {3:11.6e}\n').format(
                    Conv, c2eta1d, c2eta2d, c2eta3d)
            self.file.write(to_print)

    def treat_dissipation_rates(self, energyK_fft, energyA_fft,
                                CharneyPE_fft):
        """Compute and save dissipation rates."""

        f_d, f_d_hypo = self.sim.compute_freq_diss()

        dico_eps = self.compute_dissipation_rates(
            f_d, f_d_hypo,
            energyK_fft, energyA_fft, CharneyPE_fft)

        self.save_dissipation_rates(dico_eps)

    def compute_dissipation_rates(
            self, f_d, f_d_hypo,
            energyK_fft, energyA_fft, CharneyPE_fft):

        epsK = self.sum_wavenumbers(f_d*2*energyK_fft)
        epsK_hypo = self.sum_wavenumbers(f_d_hypo*2*energyK_fft)
        epsA = self.sum_wavenumbers(f_d*2*energyA_fft)
        epsA_hypo = self.sum_wavenumbers(f_d_hypo*2*energyA_fft)
        epsCPE = self.sum_wavenumbers(f_d*2*CharneyPE_fft)
        epsCPE_hypo = self.sum_wavenumbers(f_d_hypo*2*CharneyPE_fft)

        dico_eps = {'epsK': epsK,
                    'epsK_hypo': epsK_hypo,
                    'epsA': epsA,
                    'epsA_hypo': epsA_hypo,
                    'epsCPE': epsCPE,
                    'epsCPE_hypo': epsCPE_hypo}
        return dico_eps

    def save_dissipation_rates(self, dico_eps):
        epsK = dico_eps['epsK']
        epsK_hypo = dico_eps['epsK_hypo']
        epsA = dico_eps['epsA']
        epsA_hypo = dico_eps['epsA_hypo']
        epsCPE = dico_eps['epsCPE']
        epsCPE_hypo = dico_eps['epsCPE_hypo']

        if mpi.rank == 0:
            epsK_tot = epsK+epsK_hypo
            epsA_tot = epsA+epsA_hypo

            to_print = (
'epsK   = {0:11.6e} ; epsK_hypo  = {1:11.6e} ; epsK_tot  = {2:11.6e} \n'
'epsA   = {3:11.6e} ; epsA_hypo  = {4:11.6e} ; epsA_tot  = {5:11.6e} \n'
'epsCPE = {6:11.6e} ; epsCPEhypo = {7:11.6e} ; epsCPEtot = {8:11.6e} \n'
).format(epsK,   epsK_hypo,   epsK_tot,
         epsA,   epsA_hypo,   epsA_tot,
         epsCPE, epsCPE_hypo, epsCPE+epsCPE_hypo)
            self.file.write(to_print)

            if self.has_to_plot:
                tsim = self.sim.time_stepping.t
                self.axe_b.plot(tsim, epsK_tot+epsA_tot, 'k.')

    def treat_forcing(self):
        """Save forcing injection rates."""
        state_fft = self.sim.state.state_fft
        ux_fft = state_fft.get_var('ux_fft')
        uy_fft = state_fft.get_var('uy_fft')
        eta_fft = state_fft.get_var('eta_fft')

        forcing_fft = self.sim.forcing.get_forcing()
        Fx_fft = forcing_fft.get_var('ux_fft')
        Fy_fft = forcing_fft.get_var('uy_fft')
        Feta_fft = forcing_fft.get_var('eta_fft')

        deltat = self.sim.time_stepping.deltat

        PK1_fft = (
            inner_prod(ux_fft, Fx_fft)
            + inner_prod(uy_fft, Fy_fft)
            )
        PK2_fft = deltat/2*(abs(Fx_fft)**2 + abs(Fy_fft)**2)

        PK1 = self.sum_wavenumbers(PK1_fft)
        PK2 = self.sum_wavenumbers(PK2_fft)

        PA1_fft = self.c2*inner_prod(eta_fft, Feta_fft)
        PA2_fft = deltat/2*self.c2*(abs(Feta_fft)**2)

        PA1 = self.sum_wavenumbers(PA1_fft)
        PA2 = self.sum_wavenumbers(PA2_fft)

        if mpi.rank == 0:

            PK_tot = PK1+PK2
            PA_tot = PA1+PA2
            to_print = (
'PK1    = {0:11.6e} ; PK2        = {1:11.6e} ; PK_tot    = {2:11.6e} \n'
'PA1    = {3:11.6e} ; PA2        = {4:11.6e} ; PA_tot    = {5:11.6e} \n'
).format(PK1, PK2, PK_tot, PA1, PA2, PA_tot)

            self.file.write(to_print)

        if self.has_to_plot and mpi.rank == 0:
            tsim = self.sim.time_stepping.t
            self.axe_b.plot(tsim, PK_tot+PA_tot, 'c.')

    def load(self):
        dico_results = {'name_solver': self.output.name_solver}

        file_means = open(self.path_file)
        lines = file_means.readlines()

        lines_t = []
        lines_E = []
        lines_EK = []
        lines_epsK = []
        lines_epsA = []
        lines_epsCPE = []

        lines_epsK = []

        lines_PK = []
        lines_PA = []
        lines_etaskew = []
        lines_rotskew = []
        lines_Conv = []

        for il, line in enumerate(lines):
            if line[0:6] == 'time =':
                lines_t.append(line)
            if line[0:8] == 'E      =':
                lines_E.append(line)
            if line[0:8] == 'EK     =':
                lines_EK.append(line)
            if line[0:8] == 'epsK   =':
                lines_epsK.append(line)
            if line[0:8] == 'epsA   =':
                lines_epsA.append(line)
            if line[0:8] == 'epsCPE =':
                lines_epsCPE.append(line)
            if line[0:8] == 'PK1    =':
                lines_PK.append(line)
            if line[0:8] == 'PA1    =':
                lines_PA.append(line)
            if line.startswith('eta skew ='):
                lines_etaskew.append(line)
            if line.startswith('rot skew ='):
                lines_rotskew.append(line)
            if line.startswith('Conv ='):
                lines_Conv.append(line)

        nt = len(lines_t)
        if nt > 1:
            nt -= 1

        t = np.empty(nt)

        E = np.empty(nt)
        CPE = np.empty(nt)
        EK = np.empty(nt)
        EA = np.empty(nt)
        EKr = np.empty(nt)
        epsK = np.empty(nt)
        epsK_hypo = np.empty(nt)
        epsK_tot = np.empty(nt)
        epsA = np.empty(nt)
        epsA_hypo = np.empty(nt)
        epsA_tot = np.empty(nt)
        epsCPE = np.empty(nt)
        epsCPE_hypo = np.empty(nt)
        epsCPE_tot = np.empty(nt)

        if len(lines_PK) == len(lines_t):
            PK1 = np.empty(nt)
            PK2 = np.empty(nt)
            PK_tot = np.empty(nt)
            PA1 = np.empty(nt)
            PA2 = np.empty(nt)
            PA_tot = np.empty(nt)

        if len(lines_rotskew) == len(lines_t):
            skew_eta = np.empty(nt)
            kurt_eta = np.empty(nt)
            skew_rot = np.empty(nt)
            kurt_rot = np.empty(nt)

        if len(lines_Conv) == len(lines_t):
            Conv = np.empty(nt)
            c2eta1d = np.empty(nt)
            c2eta2d = np.empty(nt)
            c2eta3d = np.empty(nt)

        for il in xrange(nt):
            line = lines_t[il]
            words = line.split()
            t[il] = float(words[2])

            line = lines_E[il]
            words = line.split()
            E[il] = float(words[2])
            CPE[il] = float(words[6])

            line = lines_EK[il]
            words = line.split()
            EK[il] = float(words[2])
            EA[il] = float(words[6])
            EKr[il] = float(words[10])

            line = lines_epsK[il]
            words = line.split()
            epsK[il] = float(words[2])
            epsK_hypo[il] = float(words[6])
            epsK_tot[il] = float(words[10])

            line = lines_epsA[il]
            words = line.split()
            epsA[il] = float(words[2])
            epsA_hypo[il] = float(words[6])
            epsA_tot[il] = float(words[10])

            line = lines_epsCPE[il]
            words = line.split()
            epsCPE[il] = float(words[2])
            epsCPE_hypo[il] = float(words[6])
            epsCPE_tot[il] = float(words[10])


            if len(lines_PK) == len(lines_t):
                line = lines_PK[il]
                words = line.split()
                PK1[il] = float(words[2])
                PK2[il] = float(words[6])
                PK_tot[il] = float(words[10])

                line = lines_PA[il]
                words = line.split()
                PA1[il] = float(words[2])
                PA2[il] = float(words[6])
                PA_tot[il] = float(words[10])

            if len(lines_rotskew) == len(lines_t):
                line = lines_etaskew[il]
                words = line.split()
                skew_eta[il] = float(words[3])
                kurt_eta[il] = float(words[7])

                line = lines_rotskew[il]
                words = line.split()
                skew_rot[il] = float(words[3])
                kurt_rot[il] = float(words[7])

            if len(lines_Conv) == len(lines_t):
                line = lines_Conv[il]
                words = line.split()
                Conv[il] = float(words[2])
                c2eta1d[il] = float(words[6])
                c2eta2d[il] = float(words[10])
                c2eta3d[il] = float(words[14])

        dico_results['t'] = t
        dico_results['E'] = E
        dico_results['CPE'] = CPE

        dico_results['EK'] = EK
        dico_results['EA'] = EA
        dico_results['EKr'] = EKr

        dico_results['epsK'] = epsK
        dico_results['epsK_hypo'] = epsK_hypo
        dico_results['epsK_tot'] = epsK_tot

        dico_results['epsA'] = epsA
        dico_results['epsA_hypo'] = epsA_hypo
        dico_results['epsA_tot'] = epsA_tot

        dico_results['epsCPE'] = epsCPE
        dico_results['epsCPE_hypo'] = epsCPE_hypo
        dico_results['epsCPE_tot'] = epsCPE_tot

        if len(lines_PK) == len(lines_t):
            dico_results['PK1'] = PK1
            dico_results['PK2'] = PK2
            dico_results['PK_tot'] = PK_tot
            dico_results['PA1'] = PA1
            dico_results['PA2'] = PA2
            dico_results['PA_tot'] = PA_tot

        if len(lines_rotskew) == len(lines_t):
            dico_results['skew_eta'] = skew_eta
            dico_results['kurt_eta'] = kurt_eta
            dico_results['skew_rot'] = skew_rot
            dico_results['kurt_rot'] = kurt_rot

        if len(lines_Conv) == len(lines_t):
            dico_results['Conv'] = Conv
            dico_results['c2eta1d'] = c2eta1d
            dico_results['c2eta2d'] = c2eta2d
            dico_results['c2eta3d'] = c2eta3d

        return dico_results

    def plot(self):
        dico_results = self.load()

        t = dico_results['t']

        E = dico_results['E']
        CPE = dico_results['CPE']

        EK = dico_results['EK']
        EA = dico_results['EA']
        EKr = dico_results['EKr']

        epsK = dico_results['epsK']
        epsK_hypo = dico_results['epsK_hypo']
        epsK_tot = dico_results['epsK_tot']

        epsA = dico_results['epsA']
        epsA_hypo = dico_results['epsA_hypo']
        epsA_tot = dico_results['epsA_tot']

        epsE      = epsK      + epsA
        epsE_hypo = epsK_hypo + epsA_hypo
        epsE_tot  = epsK_tot  + epsA_tot

        epsCPE = dico_results['epsCPE']
        epsCPE_hypo = dico_results['epsCPE_hypo']
        epsCPE_tot = dico_results['epsCPE_tot']

        if 'PK_tot' in dico_results:
            PK_tot = dico_results['PK_tot']
            PA_tot = dico_results['PA_tot']
            P_tot = PK_tot + PA_tot

        width_axe = 0.85
        height_axe = 0.37
        x_left_axe = 0.12
        z_bottom_axe = 0.56

        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel('t')
        ax1.set_ylabel('$2E(t)/c^2$')
        title = ('mean energy, solver ' + self.output.name_solver +
                 ', nh = {0:5d}'.format(self.nx) +
                 ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f))
        ax1.set_title(title)
        ax1.hold(True)
        norm = self.c2/2
        ax1.plot(t, E/norm, 'k', linewidth=2)
        ax1.plot(t, EK/norm, 'r', linewidth=1)
        ax1.plot(t, EA/norm, 'b', linewidth=1)
        ax1.plot(t, EKr/norm, 'r--', linewidth=1)
        ax1.plot(t, (EK-EKr)/norm, 'r:', linewidth=1)

        z_bottom_axe = 0.07
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel('t')
        ax2.set_ylabel('Charney PE(t)')
        title = ('mean Charney PE(t)')
        ax2.set_title(title)
        ax2.hold(True)
        ax2.plot(t, CPE, 'k', linewidth=2)

        z_bottom_axe = 0.56
        size_axe[1] = z_bottom_axe
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel('t')
        ax1.set_ylabel('$P_E(t)$, $\epsilon(t)$')
        title = ('forcing and dissipation, solver ' + self.output.name_solver +
                 ', nh = {0:5d}'.format(self.nx) +
                 ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f))
        ax1.set_title(title)
        ax1.hold(True)
        if 'PK_tot' in dico_results:
            (l_P_tot,) = ax1.plot(t, P_tot, 'c', linewidth=2)
            l_P_tot.set_label('$P_{tot}$')

        (l_epsE,) = ax1.plot(t, epsE, 'k--', linewidth=2)
        (l_epsE_hypo,) = ax1.plot(t, epsE_hypo, 'g', linewidth=2)
        (l_epsE_tot,) = ax1.plot(t, epsE_tot, 'k', linewidth=2)

        l_epsE.set_label('$\epsilon$')
        l_epsE_hypo.set_label('$\epsilon_{hypo}$')
        l_epsE_tot.set_label('$\epsilon_{tot}$')

        ax1.legend(loc=2)

        z_bottom_axe = 0.07
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel('t')
        ax2.set_ylabel('$\epsilon$ Charney PE(t)')
        title = ('dissipation Charney PE')
        ax2.set_title(title)
        ax2.hold(True)
        ax2.plot(t, epsCPE, 'k--', linewidth=2)
        ax2.plot(t, epsCPE_hypo, 'g', linewidth=2)
        ax2.plot(t, epsCPE_tot, 'r', linewidth=2)

#         skew_eta = dico_results['skew_eta']
#         kurt_eta = dico_results['kurt_eta']
#         skew_rot = dico_results['skew_rot']
#         kurt_rot = dico_results['kurt_rot']

#         fig, ax1 = self.output.figure_axe()

#         title = ('skewness and kurtosis, solver '+self.output.name_solver+
# ', nh = {0:5d}'.format(self.nx)+
# ', c2 = {0:.4g}, f = {1:.4g}'.format(self.c2, self.f)
# )
#         ax1.set_title(title)
#         ax2.set_xlabel('t')

#         ax1.plot(t, skew_eta, 'b', linewidth=2)
#         ax1.plot(t, kurt_eta, 'b--', linewidth=2)
#         ax1.plot(t, skew_rot, 'r', linewidth=2)
#         ax1.plot(t, kurt_rot, 'r--', linewidth=2)

    def plot_rates(self, keys='E'):
        """Plots the time history of the time derivative of a spatial mean,
        and also calculates the average of the same.

        Parameters
        ----------
        key : string or a list of strings

            Refers to the the spatial mean which you want to take time
            derivative of.  Legal value include:

            For ns2d ['E', 'Z']
            For sw1l ['E', 'EK', 'EA', 'EKr', 'CPE']

        Examples
        --------
        >>> plot_rates()
        >>> plot_rates('Z')
        >>> plot_rates(['E', 'Z'])
        >>> plot_rates(['E', 'EK', 'EA', 'EKr', 'CPE'])

        """

        dico_results = self.load()
        t = dico_results['t']
        dt = np.gradient(t, 1.)

        fig, axarr = plt.subplots(len(keys), sharex=True)
        i = 0
        for k in keys:
            E = dico_results[k]
            dE_dt = abs(np.gradient(E, 1.)/dt)
            dE_dt_avg = '{0:11.6e}'.format(dE_dt.mean())
            try:
                axarr[i].semilogy(t, dE_dt, label=dE_dt_avg)
                axarr[i].set_ylabel(r'$\partial_t$' + keys[i])
                axarr[i].legend()
                #axarr[i].text(0.8, 0.9, 'mean = ' + dE_dt_avg, horizontalalignment='center', verticalalignment='center',)
            except TypeError:
                axarr.semilogy(t, dE_dt, label=dE_dt_avg)
                axarr.set_ylabel(keys)
                axarr.legend()
            i += 1

        try:
            axarr[i-1].set_xlabel('t')
        except TypeError:
            axarr.set_xlabel('t')

        plt.draw()


class SpatialMeansSW1L(SpatialMeansMSW1L):
    """Handle the saving of spatial mean quantities.


    Viz. total energy, K.E., A.P.E. and Charney potential enstrophy. It also
    handles the computation of forcing and dissipation rates for sw1l solver.

    """

    def treat_dissipation_rates(self, energyK_fft, energyA_fft,
                                CharneyPE_fft):
        """Compute and save dissipation rates."""

        f_d, f_d_hypo = self.sim.compute_freq_diss()

        dico_eps = super(
            SpatialMeansSW1L, self
        ).compute_dissipation_rates(
            f_d, f_d_hypo, energyK_fft, energyA_fft, CharneyPE_fft)

        (epsKsuppl, epsKsuppl_hypo
         ) = self.compute_epsK(f_d, f_d_hypo, energyK_fft, dico_eps)

        super(SpatialMeansSW1L, self).save_dissipation_rates(dico_eps)

        if mpi.rank == 0:
            to_print = (
                'epsKsup= {0:11.6e} ; epsKshypo  = {1:11.6e} ;\n'
            ).format(epsKsuppl,   epsKsuppl_hypo)
            self.file.write(to_print)

    def compute_epsK(self, f_d, f_d_hypo,
                     energyK_fft, dico_eps):

        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')

        EKquad = 0.5*(ux**2 + uy**2)
        EKquad_fft = self.sim.oper.fft2(EKquad)

        eta_fft = self.sim.state('eta_fft')

        epsKsuppl = self.sum_wavenumbers(
            f_d*inner_prod(EKquad_fft, eta_fft))

        epsKsuppl_hypo = self.sum_wavenumbers(
            f_d_hypo*inner_prod(EKquad_fft, eta_fft)
            )

        dico_eps['epsK'] += epsKsuppl
        dico_eps['epsK_hypo'] += epsKsuppl_hypo

        return epsKsuppl, epsKsuppl_hypo

    def load(self):

        dico_results = super(SpatialMeansSW1L, self).load()

        file_means = open(self.path_file)
        lines = file_means.readlines()

        lines_epsKsuppl = []

        for il, line in enumerate(lines):
            if line.startswith('epsKsup='):
                lines_epsKsuppl.append(line)

        t = dico_results['t']
        nt = len(t)
        epsKsuppl = np.empty(nt)
        epsKsuppl_hypo = np.empty(nt)

        for il in xrange(nt):
            line = lines_epsKsuppl[il]
            words = line.split()
            epsKsuppl[il] = float(words[1])
            epsKsuppl_hypo[il] = float(words[5])

        dico_results['epsKsuppl'] = epsKsuppl
        dico_results['epsKsuppl_hypo'] = epsKsuppl_hypo

        return dico_results

    def treat_forcing(self):
        """
        Save forcing injection rates.
        """
        state = self.sim.state
        ux_fft = state('ux_fft')
        uy_fft = state('uy_fft')
        eta_fft = state('eta_fft')

        forcing = self.sim.forcing
        Fx_fft = forcing('ux_fft')
        Fy_fft = forcing('uy_fft')
        Feta_fft = forcing('eta_fft')

        deltat = self.sim.time_stepping.deltat

        PA1_fft = self.c2*inner_prod(eta_fft, Feta_fft)
        PA2_fft = deltat/2*self.c2*(abs(Feta_fft)**2)

        PA1 = self.sum_wavenumbers(PA1_fft)
        PA2 = self.sum_wavenumbers(PA2_fft)

        Fx = self.sim.oper.ifft2(Fx_fft)
        Fy = self.sim.oper.ifft2(Fy_fft)
        Feta = self.sim.oper.ifft2(Feta_fft)

        eta = self.sim.state.state_phys.get_var('eta')
        h = eta + 1.

        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')

        FetaFx_fft = self.sim.oper.fft2(Feta*Fx)
        FetaFy_fft = self.sim.oper.fft2(Feta*Fy)

        Jx_fft = self.sim.oper.fft2(h*ux)
        Jy_fft = self.sim.oper.fft2(h*uy)

        FJx_fft = self.sim.oper.fft2(h*Fx + Feta*ux)
        FJy_fft = self.sim.oper.fft2(h*Fy + Feta*uy)

        PK1_fft = 0.5*(
            inner_prod(Jx_fft, Fx_fft)
            + inner_prod(Jy_fft, Fy_fft)
            + inner_prod(ux_fft, FJx_fft)
            + inner_prod(uy_fft, FJy_fft)
            )
        PK2_fft = deltat/2*(
            0.5*(inner_prod(Fx_fft, FJx_fft)
                 + inner_prod(Fy_fft, FJy_fft)
                 )
            + inner_prod(ux_fft, FetaFx_fft)
            + inner_prod(uy_fft, FetaFy_fft)
            )

        PK1 = self.sum_wavenumbers(PK1_fft)
        PK2 = self.sum_wavenumbers(PK2_fft)

        if mpi.rank == 0:

            PK_tot = PK1+PK2
            PA_tot = PA1+PA2
            to_print = (
'PK1    = {0:11.6e} ; PK2        = {1:11.6e} ; PK_tot    = {2:11.6e} \n'
'PA1    = {3:11.6e} ; PA2        = {4:11.6e} ; PA_tot    = {5:11.6e} \n'
).format(PK1, PK2, PK_tot, PA1, PA2, PA_tot)

            self.file.write(to_print)

        if self.has_to_plot and mpi.rank == 0:
            tsim = self.sim.time_stepping.t
            self.axe_b.plot(tsim, PK_tot+PA_tot, 'c.')
