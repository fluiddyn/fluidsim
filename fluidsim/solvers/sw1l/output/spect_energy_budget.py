import numpy as np
import h5py

from fluiddyn.util import mpi

from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase, cumsum_inv, inner_prod)


class SpectralEnergyBudgetSW1l(SpectralEnergyBudgetBase):
    """Save and plot spectra."""

    def __init__(self, output):

        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super(SpectralEnergyBudgetSW1l, self).__init__(output)

    def compute(self):
        """compute spectral energy budget the one time."""
        oper = self.sim.oper

        # print_memory_usage('start function compute seb')

        ux = self.sim.state.state_phys['ux']
        uy = self.sim.state.state_phys['uy']
        eta = self.sim.state.state_phys['eta']
        h = 1.+eta

        Jx = h*ux
        Jy = h*uy
        Jx_fft = oper.fft2(Jx)
        Jy_fft = oper.fft2(Jy)
        del(Jx, Jy)

        ux_fft = self.sim.state('ux_fft')
        uy_fft = self.sim.state('uy_fft')
        eta_fft = self.sim.state('eta_fft')
        h_fft = eta_fft.copy()
        if mpi.rank == 0:
            h_fft[0, 0] = 1.

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        urx_fft, ury_fft = oper.vecfft_from_rotfft(rot_fft)
        del(rot_fft)
        urx = oper.ifft2(urx_fft)
        ury = oper.ifft2(ury_fft)

        q_fft, div_fft, a_fft = \
            self.oper.qdafft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        # if self.params.f != 0:
        #     udx_fft, udy_fft = oper.vecfft_from_divfft(div_fft)
        #     ugx_fft, ugy_fft, etag_fft = \
        #         self.oper.uxuyetafft_from_qfft(q_fft)
        #     uax_fft, uay_fft, etaa_fft = \
        #         self.oper.uxuyetafft_from_afft(a_fft)
        #     del(a_fft)
        #     # velocity influenced by linear terms
        #     u_infl_lin_x = udx_fft + uax_fft
        #     u_infl_lin_y = udy_fft + uay_fft

        udx_fft, udy_fft = oper.vecfft_from_divfft(div_fft)
        udx = oper.ifft2(udx_fft)
        udy = oper.ifft2(udy_fft)
        div = oper.ifft2(div_fft)
        del(div_fft)

        # print_memory_usage('before starting computing fluxes')


        # compute flux of Charney PE
        Fq_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, q_fft)
        transferCPE_fft = inner_prod(q_fft, Fq_fft)
        del(q_fft, Fq_fft)
        transfer2D_CPE = self.spectrum2D_from_fft(transferCPE_fft)
        del(transferCPE_fft)

#         print(
# ('sum(transfer2D_CPE) = {0:9.4e} ; sum(abs(transfer2D_CPE)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_CPE),
# np.sum(abs(transfer2D_CPE)))
# )



        px_h_fft, py_h_fft = oper.gradfft_from_fft(eta_fft)
        px_h = oper.ifft2(px_h_fft)
        py_h = oper.ifft2(py_h_fft)

        F_rh = -urx*px_h - ury*py_h
        F_dh = -udx*px_h - udy*py_h - h*div/2
        F_de = -udx*px_h - udy*py_h - eta*div/2
        del(px_h, py_h)
        F_rh_fft = oper.fft2(F_rh)
        F_dh_fft = oper.fft2(F_dh)
        F_de_fft = oper.fft2(F_de)
        del(F_rh, F_dh, F_de)
        oper.dealiasing(F_rh_fft)
        oper.dealiasing(F_dh_fft)
        oper.dealiasing(F_de_fft)

        transferEAr_fft = self.c2*inner_prod(h_fft, F_rh_fft)
        transferEPd_fft = self.c2*inner_prod(h_fft, F_dh_fft)
        transferEAd_fft = self.c2*inner_prod(eta_fft, F_de_fft)
        del(F_rh_fft, F_dh_fft, F_de_fft)

        transfer2D_EAr = self.spectrum2D_from_fft(transferEAr_fft)
        transfer2D_EPd = self.spectrum2D_from_fft(transferEPd_fft)
        transfer2D_EAd = self.spectrum2D_from_fft(transferEAd_fft)
        del(transferEAr_fft, transferEPd_fft, transferEAd_fft)

        # print_memory_usage('after transfer2D_EAr')

#         print(
# ('sum(transfer2D_EAr) = {0:9.4e} ; sum(abs(transfer2D_EAr)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_EAr),
# np.sum(abs(transfer2D_EAr)))
# )

#         print(
# ('sum(transfer2D_EAd) = {0:9.4e} ; sum(abs(transfer2D_EAd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_EAd),
# np.sum(abs(transfer2D_EAd)))
# )

        hdiv_fft = oper.fft2(h*div)
        convP_fft = self.c2/2.*inner_prod(h_fft, hdiv_fft)
        convP2D = self.spectrum2D_from_fft(convP_fft)
        del(convP_fft, h_fft, hdiv_fft)

        EP = self.c2/2*h*h
        EP_fft = oper.fft2(EP)
        del(EP, h)
        px_EP_fft, py_EP_fft = oper.gradfft_from_fft(EP_fft)
        del(EP_fft)

        convK_fft = 1./2*(
            - inner_prod(ux_fft, px_EP_fft)
            - inner_prod(uy_fft, py_EP_fft)
            - self.c2*inner_prod(Jx_fft, px_h_fft)
            - self.c2*inner_prod(Jy_fft, py_h_fft)
            )
        del(px_h_fft, py_h_fft, px_EP_fft, py_EP_fft)
        convK2D = self.spectrum2D_from_fft(convK_fft)
        del(convK_fft)

        # print_memory_usage('after convK2D')

#         print(
# ('sum(convP2D-convK2D)*deltakh = {0:9.4e}, sum(convP2D)*deltakh = {1:9.4e}'
# ).format(
# np.sum(convP2D-convK2D)*self.oper.deltakh,
# np.sum(convP2D)*self.oper.deltakh
# )
# )

#         print(
# ('                                           sum(convK2D)*deltakh = {0:9.4e}'
# ).format(
# np.sum(convK2D)*self.oper.deltakh
# )
# )


        Fxrd_fft, Fxdd_fft = self.fnonlinfft_from_uruddivfunc(
            urx, ury, udx, udy, div, udx_fft, udx)
        Fyrd_fft, Fydd_fft = self.fnonlinfft_from_uruddivfunc(
            urx, ury, udx, udy, div, udy_fft, udy)
        Fxrr_fft, Fxdr_fft = self.fnonlinfft_from_uruddivfunc(
            urx, ury, udx, udy, div, urx_fft, urx)
        Fyrr_fft, Fydr_fft = self.fnonlinfft_from_uruddivfunc(
            urx, ury, udx, udy, div, ury_fft, ury)


        transferErrr_fft = (  inner_prod(urx_fft, Fxrr_fft)
                            + inner_prod(ury_fft, Fyrr_fft)
                            )
        transfer2D_Errr = self.spectrum2D_from_fft(transferErrr_fft)
        del(transferErrr_fft)
#         print(
# ('sum(transfer2D_Errr) = {0:9.4e} ; sum(abs(transfer2D_Errr)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Errr),
# np.sum(abs(transfer2D_Errr)))
# )

        transferEdrd_fft = (  inner_prod(udx_fft, Fxrd_fft)
                            + inner_prod(udy_fft, Fyrd_fft)
                            )
        transfer2D_Edrd = self.spectrum2D_from_fft(transferEdrd_fft)
        del(transferEdrd_fft)
#         print(
# ('sum(transfer2D_Edrd) = {0:9.4e} ; sum(abs(transfer2D_Edrd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Edrd),
# np.sum(abs(transfer2D_Edrd)))
# )
        Clfromqq = (  inner_prod(udx_fft, Fxrr_fft)
                    + inner_prod(udy_fft, Fyrr_fft)
                      )
        transferEdrr_rrd_fft = (  Clfromqq
                                + inner_prod(urx_fft, Fxrd_fft)
                                + inner_prod(ury_fft, Fyrd_fft)
                                  )
        Clfromqq = self.spectrum2D_from_fft(Clfromqq)
        transfer2D_Edrr_rrd = self.spectrum2D_from_fft(transferEdrr_rrd_fft)
        del(transferEdrr_rrd_fft)
#         print(
# ('sum(transfer2D_Edrr_rrd) = {0:9.4e} ; '
# 'sum(abs(transfer2D_Edrr_rrd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Edrr_rrd),
# np.sum(abs(transfer2D_Edrr_rrd)))
# )



        transferErdr_fft = (  inner_prod(urx_fft, Fxdr_fft)
                            + inner_prod(ury_fft, Fydr_fft)
                            )
        transfer2D_Erdr = self.spectrum2D_from_fft(transferErdr_fft)
        del(transferErdr_fft)
#         print(
# ('sum(transfer2D_Erdr) = {0:9.4e} ; sum(abs(transfer2D_Erdr)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Erdr),
# np.sum(abs(transfer2D_Erdr)))
# )

        transferEddd_fft = (  inner_prod(udx_fft, Fxdd_fft)
                            + inner_prod(udy_fft, Fydd_fft)
                            )
        transfer2D_Eddd = self.spectrum2D_from_fft(transferEddd_fft)
        del(transferEddd_fft)
#         print(
# ('sum(transfer2D_Eddd) = {0:9.4e} ; sum(abs(transfer2D_Eddd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Eddd),
# np.sum(abs(transfer2D_Eddd)))
# )

        Cqfromll = (  inner_prod(urx_fft, Fxdd_fft)
                     + inner_prod(ury_fft, Fydd_fft)
                       )

        transferEddr_rdd_fft = (  Cqfromll
                                + inner_prod(udx_fft, Fxdr_fft)
                                + inner_prod(udy_fft, Fydr_fft)
                                  )

        Cqfromll = self.spectrum2D_from_fft(Cqfromll)
        transfer2D_Eddr_rdd = self.spectrum2D_from_fft(transferEddr_rdd_fft)
        del(transferEddr_rdd_fft)
#         print(
# ('sum(transfer2D_Eddr_rdd) = {0:9.4e} ; '
# 'sum(abs(transfer2D_Eddr_rdd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Eddr_rdd),
# np.sum(abs(transfer2D_Eddr_rdd)))
# )


        Fx_ru_fft = Fxrr_fft + Fxrd_fft
        del(Fxrr_fft,  Fxrd_fft)
        Fy_ru_fft = Fyrr_fft + Fyrd_fft
        del(Fyrr_fft, Fyrd_fft)
        Fx_du_fft = Fxdr_fft + Fxdd_fft
        del(Fxdr_fft,  Fxdd_fft)
        Fy_du_fft = Fydr_fft + Fydd_fft
        del(Fydr_fft, Fydd_fft)


        # print_memory_usage('after Fy_du_fft')


        etaux = eta*ux
        etauy = eta*uy

        etaux_fft = oper.fft2(etaux)
        etauy_fft = oper.fft2(etauy)
        del(etaux, etauy)

        px_etaux_fft, py_etaux_fft = oper.gradfft_from_fft(etaux_fft)
        px_etauy_fft, py_etauy_fft = oper.gradfft_from_fft(etauy_fft)

        px_etaux = oper.ifft2(px_etaux_fft)
        del(px_etaux_fft)
        py_etaux = oper.ifft2(py_etaux_fft)
        del(py_etaux_fft)
        px_etauy = oper.ifft2(px_etauy_fft)
        del(px_etauy_fft)
        py_etauy = oper.ifft2(py_etauy_fft)
        del(py_etauy_fft)

        Fx_reu = -urx*px_etaux - ury*py_etaux
        Fx_reu_fft = oper.fft2(Fx_reu)
        del(Fx_reu)

        Fx_deu = -udx*px_etaux - udy*py_etaux - 0.5*div*eta*ux
        del(px_etaux, py_etaux)
        Fx_deu_fft = oper.fft2(Fx_deu)
        del(Fx_deu)

        Fy_reu = -urx*px_etauy - ury*py_etauy
        Fy_reu_fft = oper.fft2(Fy_reu)
        del(Fy_reu)

        Fy_deu = -udx*px_etauy - udy*py_etauy - 0.5*div*eta*uy
        del(px_etauy, py_etauy)
        Fy_deu_fft = oper.fft2(Fy_deu)
        del(Fy_deu)



        transferEureu_fft = 0.5*(
              inner_prod(ux_fft, Fx_reu_fft)
            + inner_prod(uy_fft, Fy_reu_fft)
            + inner_prod(etaux_fft, Fx_ru_fft)
            + inner_prod(etauy_fft, Fy_ru_fft)
            )
        del(Fx_reu_fft, Fy_reu_fft, Fx_ru_fft, Fy_ru_fft)

        transfer2D_Eureu = self.spectrum2D_from_fft(transferEureu_fft)
#         print(
# ('sum(transferEureu_fft) = {0:9.4e} ; '
# 'sum(abs(transferEureu_fft)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Eureu),
# np.sum(abs(transfer2D_Eureu)))
# )

        transferEudeu_fft = 0.5*(
              inner_prod(ux_fft, Fx_deu_fft)
            + inner_prod(uy_fft, Fy_deu_fft)
            + inner_prod(etaux_fft, Fx_du_fft)
            + inner_prod(etauy_fft, Fy_du_fft)
            )
        del(Fx_deu_fft, Fy_deu_fft, Fx_du_fft, Fy_du_fft)
        del(etaux_fft, etauy_fft)

        transfer2D_Eudeu = self.spectrum2D_from_fft(transferEudeu_fft)
        del(transferEudeu_fft)
#         print(
# ('sum(transferEudeu_fft) = {0:9.4e} ; '
# 'sum(abs(transferEudeu_fft)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Eudeu),
# np.sum(abs(transfer2D_Eudeu)))
# )

        transfer2D_EKr = (transfer2D_Errr + transfer2D_Edrd
                          + transfer2D_Edrr_rrd
                          + transfer2D_Eureu
                          )
        transfer2D_EKd = (transfer2D_Erdr + transfer2D_Eddd
                          + transfer2D_Eddr_rdd
                          + transfer2D_Eudeu
                          )

        # print_memory_usage('end of function compute seb')


        dico_results = {
            'transfer2D_EAr': transfer2D_EAr,
            'transfer2D_EAd': transfer2D_EAd,
            'transfer2D_EPd': transfer2D_EPd,
            'transfer2D_EKr': transfer2D_EKr,
            'transfer2D_EKd': transfer2D_EKd,
            'transfer2D_Errr': transfer2D_Errr,
            'transfer2D_Edrd': transfer2D_Edrd,
            'Clfromqq': Clfromqq,
            'transfer2D_Edrr_rrd': transfer2D_Edrr_rrd,
            'transfer2D_Erdr': transfer2D_Erdr,
            'transfer2D_Eddd': transfer2D_Eddd,
            'Cqfromll': Cqfromll,
            'transfer2D_Eddr_rdd': transfer2D_Eddr_rdd,
            'transfer2D_Eureu': transfer2D_Eureu,
            'transfer2D_Eudeu': transfer2D_Eudeu,
            'convP2D': convP2D,
            'convK2D': convK2D,
            'transfer2D_CPE': transfer2D_CPE,
}
        return dico_results





    def _online_plot(self, dico_results):

        transfer2D_CPE = dico_results['transfer2D_CPE']
        transfer2D_EKr = dico_results['transfer2D_EKr']
        transfer2D_EKd = dico_results['transfer2D_EKd']
        transfer2D_EK = transfer2D_EKr + transfer2D_EKd
        transfer2D_EAr = dico_results['transfer2D_EAr']
        transfer2D_EAd = dico_results['transfer2D_EAd']
        transfer2D_EA = transfer2D_EAr + transfer2D_EAd
        convP2D = dico_results['convP2D']
        convK2D = dico_results['convK2D']
        khE = self.oper.khE
        PiCPE = cumsum_inv(transfer2D_CPE)*self.oper.deltakh
        PiEK = cumsum_inv(transfer2D_EK)*self.oper.deltakh
        PiEA = cumsum_inv(transfer2D_EA)*self.oper.deltakh
        CCP = cumsum_inv(convP2D)*self.oper.deltakh
        CCK = cumsum_inv(convK2D)*self.oper.deltakh

        self.axe_a.plot(khE+khE[1], PiEK, 'r')
        self.axe_a.plot(khE+khE[1], PiEA, 'b')
        self.axe_a.plot(khE+khE[1], CCP, 'y')
        self.axe_a.plot(khE+khE[1], CCK, 'y--')

        self.axe_b.plot(khE+khE[1], PiCPE, 'g')




    def plot(self, tmin=0, tmax=1000, delta_t=2):

        f = h5py.File(self.path_file, 'r')

        dset_times = f['times']
        times = dset_times[...]
        # nt = len(times)

        dset_khE = f['khE']
        khE = dset_khE[...]

        dset_transfer2D_EKr = f['transfer2D_EKr']
        dset_transfer2D_EKd = f['transfer2D_EKd']
        dset_transfer2D_EAr = f['transfer2D_EAr']
        dset_transfer2D_EAd = f['transfer2D_EAd']
        dset_transfer2D_EPd = f['transfer2D_EPd']
        dset_convP2D = f['convP2D']
        dset_convK2D = f['convK2D']
        dset_transfer2D_CPE = f['transfer2D_CPE']

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot = 1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]


        to_print = 'plot(tmin={0}, tmax={1}, delta_t={2:.2f})'.format(
            tmin, tmax, delta_t)
        print(to_print)

        to_print = '''plot fluxes 2D
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
        print(to_print)




        x_left_axe = 0.12
        z_bottom_axe = 0.36
        width_axe = 0.85
        height_axe = 0.57

        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('transfers')
        title = ('energy flux, solver '+self.output.name_solver+
', nh = {0:5d}'.format(self.nx)+
', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
)
        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('linear')


        khE = khE+1

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot, delta_i_plot):
                transferEKr = dset_transfer2D_EKr[it]
                transferEAr = dset_transfer2D_EAr[it]
                PiEKr = cumsum_inv(transferEKr)*self.oper.deltakh
                PiEAr = cumsum_inv(transferEAr)*self.oper.deltakh
                PiE = PiEKr + PiEAr
                ax1.plot(khE, PiE, 'k', linewidth=1)

                convK = dset_convK2D[it]
                CCK = cumsum_inv(convK)*self.oper.deltakh
                ax1.plot(khE, CCK, 'y', linewidth=1)

                convP = dset_convP2D[it]
                CCP = cumsum_inv(convP)*self.oper.deltakh
                ax1.plot(khE, CCP, 'y--', linewidth=1)

                # print(convK.sum()*self.oper.deltakh,
                #       convP.sum()*self.oper.deltakh,
                #       CCP[0], CCK[0])




        transferEKr = dset_transfer2D_EKr[imin_plot:imax_plot].mean(0)
        transferEKd = dset_transfer2D_EKd[imin_plot:imax_plot].mean(0)
        transferEAr = dset_transfer2D_EAr[imin_plot:imax_plot].mean(0)
        transferEAd = dset_transfer2D_EAd[imin_plot:imax_plot].mean(0)
        transferEPd = dset_transfer2D_EPd[imin_plot:imax_plot].mean(0)


        PiEKr = cumsum_inv(transferEKr)*self.oper.deltakh
        PiEKd = cumsum_inv(transferEKd)*self.oper.deltakh
        PiEAr = cumsum_inv(transferEAr)*self.oper.deltakh
        PiEAd = cumsum_inv(transferEAd)*self.oper.deltakh
        PiEPd = cumsum_inv(transferEPd)*self.oper.deltakh

        PiEK = PiEKr + PiEKd
        PiEA = PiEAr + PiEAd
        PiE = PiEK + PiEA

        ax1.plot(khE, PiE, 'k', linewidth=2)
        ax1.plot(khE, PiEK, 'r', linewidth=2)
        ax1.plot(khE, PiEA, 'b', linewidth=2)



        ax1.plot(khE, PiEKr, 'r--', linewidth=2)
        ax1.plot(khE, PiEKd, 'r:', linewidth=2)


        ax1.plot(khE, PiEAr, 'b--', linewidth=2)
        ax1.plot(khE, PiEAd, 'b:', linewidth=2)
        # ax1.plot(khE, PiEPd, 'c:', linewidth=1)





        convP = dset_convP2D[imin_plot:imax_plot].mean(0)
        convK = dset_convK2D[imin_plot:imax_plot].mean(0)

        CCP = cumsum_inv(convP)*self.oper.deltakh
        CCK = cumsum_inv(convK)*self.oper.deltakh

        ax1.plot(khE, CCP, 'y--', linewidth=2)
        ax1.plot(khE, CCK, 'y', linewidth=2)

#         print(convK.sum()*self.oper.deltakh,
#               convP.sum()*self.oper.deltakh,
#               CCP[0], CCK[0],
#               CCP[1], CCK[1]
# )



        dset_transfer2D_Errr = f['transfer2D_Errr']
        dset_transfer2D_Edrd = f['transfer2D_Edrd']
        dset_transfer2D_Edrr_rrd = f['transfer2D_Edrr_rrd']

        transferEdrr_rrd = \
            dset_transfer2D_Edrr_rrd[imin_plot:imax_plot].mean(0)
        transferErrr = dset_transfer2D_Errr[imin_plot:imax_plot].mean(0)
        transferEdrd = dset_transfer2D_Edrd[imin_plot:imax_plot].mean(0)

        Pi_drr_rrd = cumsum_inv(transferEdrr_rrd)*self.oper.deltakh
        Pi_rrr = cumsum_inv(transferErrr)*self.oper.deltakh
        Pi_drd = cumsum_inv(transferEdrd)*self.oper.deltakh

        ax1.plot(khE, Pi_drr_rrd, 'm:', linewidth=1)
        ax1.plot(khE, Pi_rrr, 'm--', linewidth=1)
        ax1.plot(khE, Pi_drd, 'm-.', linewidth=1)



        dset_transfer2D_Eddd = f['transfer2D_Eddd']
        dset_transfer2D_Erdr = f['transfer2D_Erdr']
        dset_transfer2D_Eddr_rdd = f['transfer2D_Eddr_rdd']
        dset_transfer2D_Eudeu = f['transfer2D_Eudeu']


        transferEddr_rdd = \
            dset_transfer2D_Eddr_rdd[imin_plot:imax_plot].mean(0)
        transferEddd = dset_transfer2D_Eddd[imin_plot:imax_plot].mean(0)
        transferErdr = dset_transfer2D_Erdr[imin_plot:imax_plot].mean(0)

        transferEudeu = dset_transfer2D_Eudeu[imin_plot:imax_plot].mean(0)

        Pi_ddr_rdd = cumsum_inv(transferEddr_rdd)*self.oper.deltakh
        Pi_ddd = cumsum_inv(transferEddd)*self.oper.deltakh
        Pi_rdr = cumsum_inv(transferErdr)*self.oper.deltakh

        Pi_udeu = cumsum_inv(transferEudeu)*self.oper.deltakh


        ax1.plot(khE, Pi_ddr_rdd, 'c:', linewidth=1)
        ax1.plot(khE, Pi_ddd, 'c--', linewidth=1)
        ax1.plot(khE, Pi_rdr, 'c-.', linewidth=1)

        ax1.plot(khE, Pi_udeu, 'g', linewidth=1)





        z_bottom_axe = 0.07
        height_axe = 0.17
        size_axe[1] = z_bottom_axe
        size_axe[3] = height_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel('$k_h$')
        ax2.set_ylabel('transfers')
        title = ('Charney PE flux')
        ax2.set_title(title)
        ax2.hold(True)
        ax2.set_xscale('log')
        ax2.set_yscale('linear')

        if delta_t != 0.:
            for it in xrange(imin_plot,imax_plot+1,delta_i_plot):
                transferCPE = dset_transfer2D_CPE[it]
                PiCPE = cumsum_inv(transferCPE)*self.oper.deltakh
                ax2.plot(khE, PiCPE, 'g', linewidth=1)

        transferCPE = dset_transfer2D_CPE[imin_plot:imax_plot].mean(0)
        PiCPE = cumsum_inv(transferCPE)*self.oper.deltakh

        ax2.plot(khE, PiCPE, 'm', linewidth=2)

        f.close()




class SpectralEnergyBudgetMSW1l(SpectralEnergyBudgetSW1l):
    """Save and plot spectra."""




    def compute(self):
        """compute spectral energy budget the one time."""
        oper = self.sim.oper

        # ux = self.sim.state.state_phys['ux']
        # uy = self.sim.state.state_phys['uy']

        ux_fft = self.sim.state.state_fft['ux_fft']
        uy_fft = self.sim.state.state_fft['uy_fft']
        eta_fft = self.sim.state.state_fft['eta_fft']

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        urx_fft, ury_fft = oper.vecfft_from_rotfft(rot_fft)
        del(rot_fft)
        urx = oper.ifft2(urx_fft)
        ury = oper.ifft2(ury_fft)

        q_fft, div_fft, a_fft = \
            self.oper.qdafft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        udx_fft, udy_fft = oper.vecfft_from_divfft(div_fft)

        if self.params.f != 0:
            ugx_fft, ugy_fft, etag_fft = \
                self.oper.uxuyetafft_from_qfft(q_fft)
            uax_fft, uay_fft, etaa_fft = \
                self.oper.uxuyetafft_from_afft(a_fft)
            # velocity influenced by linear terms
            u_infl_lin_x = udx_fft + uax_fft
            u_infl_lin_y = udy_fft + uay_fft




        # compute flux of Charney PE
        Fq_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, q_fft)

        transferCPE_fft = inner_prod(q_fft, Fq_fft)
        del(q_fft, Fq_fft)

        transfer2D_CPE = self.spectrum2D_from_fft(transferCPE_fft)
        del(transferCPE_fft)

#         print(
# ('sum(transfer2D_CPE) = {0:9.4e} ; sum(abs(transfer2D_CPE)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_CPE),
# np.sum(abs(transfer2D_CPE)))
# )


        Feta_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, eta_fft)
        transferEA_fft = self.c2*inner_prod(eta_fft, Feta_fft)
        del(Feta_fft)
        transfer2D_EA = self.spectrum2D_from_fft(transferEA_fft)
        del(transferEA_fft)

#         print(
# ('sum(transfer2D_EA) = {0:9.4e} ; sum(abs(transfer2D_EA)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_EA),
# np.sum(abs(transfer2D_EA)))
# )


        convA_fft = self.c2*inner_prod(eta_fft, div_fft)
        convA2D = self.spectrum2D_from_fft(convA_fft)
        del(convA_fft)


        Fxrr_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, urx_fft)
        Fyrr_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, ury_fft)

        Fxrd_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, udx_fft)
        Fyrd_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, udy_fft)

        transferErrr_fft = (  inner_prod(urx_fft, Fxrr_fft)
                            + inner_prod(ury_fft, Fyrr_fft)
                          )
        transfer2D_Errr = self.spectrum2D_from_fft(transferErrr_fft)
        del(transferErrr_fft)
#         print(
# ('sum(transfer2D_Errr) = {0:9.4e} ; sum(abs(transfer2D_Errr)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Errr),
# np.sum(abs(transfer2D_Errr)))
# )

        transferEdrd_fft = (  inner_prod(udx_fft, Fxrd_fft)
                            + inner_prod(udy_fft, Fyrd_fft)
                            )
        transfer2D_Edrd = self.spectrum2D_from_fft(transferEdrd_fft)
        del(transferEdrd_fft)
#         print(
# ('sum(transfer2D_Edrd) = {0:9.4e} ; sum(abs(transfer2D_Edrd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Edrd),
# np.sum(abs(transfer2D_Edrd)))
# )


        Clfromqq = (  inner_prod(udx_fft, Fxrr_fft)
                    + inner_prod(udy_fft, Fyrr_fft)
                      )
        transferEdrr_rrd_fft = (  Clfromqq
                                + inner_prod(urx_fft, Fxrd_fft)
                                + inner_prod(ury_fft, Fyrd_fft)
                                  )
        Clfromqq = self.spectrum2D_from_fft(Clfromqq)
        transfer2D_Edrr_rrd = self.spectrum2D_from_fft(transferEdrr_rrd_fft)
        del(transferEdrr_rrd_fft)
#         print(
# ('sum(transfer2D_Edrr_rrd) = {0:9.4e} ; '
# 'sum(abs(transfer2D_Edrr_rrd)) = {1:9.4e}'
# ).format(
# np.sum(transfer2D_Edrr_rrd),
# np.sum(abs(transfer2D_Edrr_rrd)))
# )

        transfer2D_EK = transfer2D_Errr + transfer2D_Edrd +transfer2D_Edrr_rrd

        dico_results = {
            'transfer2D_EK': transfer2D_EK,
            'transfer2D_Errr': transfer2D_Errr,
            'transfer2D_Edrd': transfer2D_Edrd,
            'Clfromqq': Clfromqq,
            'transfer2D_Edrr_rrd': transfer2D_Edrr_rrd,
            'transfer2D_EA': transfer2D_EA,
            'convA2D': convA2D,
            'transfer2D_CPE': transfer2D_CPE}
        return dico_results





    def _online_plot(self, dico_results):

        transfer2D_CPE = dico_results['transfer2D_CPE']
        transfer2D_EK = dico_results['transfer2D_EK']
        transfer2D_EA = dico_results['transfer2D_EA']
        convA2D = dico_results['convA2D']
        khE = self.oper.khE
        PiCPE = cumsum_inv(transfer2D_CPE)*self.oper.deltakh
        PiEK = cumsum_inv(transfer2D_EK)*self.oper.deltakh
        PiEA = cumsum_inv(transfer2D_EA)*self.oper.deltakh
        CCA = cumsum_inv(convA2D)*self.oper.deltakh
        self.axe_a.plot(khE+khE[1], PiEK, 'r')
        self.axe_a.plot(khE+khE[1], PiEA, 'b')
        self.axe_a.plot(khE+khE[1], CCA, 'y')
        self.axe_b.plot(khE+khE[1], PiCPE, 'g')


    def plot(self, tmin=0, tmax=1000, delta_t=2):

        f = h5py.File(self.path_file, 'r')

        dset_times = f['times']
        dset_khE = f['khE']
        khE = dset_khE[...]
        # khE = khE+khE[1]

        dset_transfer2D_EK = f['transfer2D_EK']
        dset_transfer2D_Errr = f['transfer2D_Errr']
        dset_transfer2D_Edrd = f['transfer2D_Edrd']
        dset_transfer2D_Edrr_rrd = f['transfer2D_Edrr_rrd']
        dset_transfer2D_EA = f['transfer2D_EA']
        dset_convA2D = f['convA2D']
        dset_transfer2D_CPE = f['transfer2D_CPE']

        times = dset_times[...]
        nt = len(times)

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0 and delta_t != 0.:
            delta_i_plot=1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]


        to_print = 'plot(tmin={0}, tmax={1}, delta_t={2:.2f})'.format(
            tmin, tmax, delta_t)
        print(to_print)

        to_print = '''plot fluxes 2D
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
        print(to_print)




        x_left_axe = 0.12
        z_bottom_axe = 0.36
        width_axe = 0.85
        height_axe = 0.57

        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel('$k_h$')
        ax1.set_ylabel('transfers')
        title = ('energy flux, solver '+self.output.name_solver+
', nh = {0:5d}'.format(self.nx)+
', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f)
)
        ax1.set_title(title)
        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('linear')

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot, delta_i_plot):
                transferEK = dset_transfer2D_EK[it]
                transferEA = dset_transfer2D_EA[it]
                PiEK = cumsum_inv(transferEK)*self.oper.deltakh
                PiEA = cumsum_inv(transferEA)*self.oper.deltakh
                PiE = PiEK + PiEA
                ax1.plot(khE, PiE, 'k', linewidth=1)

        transferEK = dset_transfer2D_EK[imin_plot:imax_plot].mean(0)
        transferEA = dset_transfer2D_EA[imin_plot:imax_plot].mean(0)
        PiEK = cumsum_inv(transferEK)*self.oper.deltakh
        PiEA = cumsum_inv(transferEA)*self.oper.deltakh
        PiE = PiEK + PiEA

        ax1.plot(khE, PiE, 'k', linewidth=2)
        ax1.plot(khE, PiEK, 'r', linewidth=2)
        ax1.plot(khE, PiEA, 'b', linewidth=2)


        transferEdrr_rrd = \
            dset_transfer2D_Edrr_rrd[imin_plot:imax_plot].mean(0)
        transferErrr = dset_transfer2D_Errr[imin_plot:imax_plot].mean(0)
        transferEdrd = dset_transfer2D_Edrd[imin_plot:imax_plot].mean(0)

        Pi_drr_rrd = cumsum_inv(transferEdrr_rrd)*self.oper.deltakh
        Pi_rrr = cumsum_inv(transferErrr)*self.oper.deltakh
        Pi_drd = cumsum_inv(transferEdrd)*self.oper.deltakh

        ax1.plot(khE, Pi_drr_rrd, 'm:', linewidth=1)
        ax1.plot(khE, Pi_rrr, 'm--', linewidth=1)
        ax1.plot(khE, Pi_drd, 'm-.', linewidth=1)




        convA2D = dset_convA2D[imin_plot:imax_plot].mean(0)
        CCA = cumsum_inv(convA2D)*self.oper.deltakh

        ax1.plot(khE+khE[1], CCA, 'y', linewidth=2)

        z_bottom_axe = 0.07
        height_axe = 0.17
        size_axe[1] = z_bottom_axe
        size_axe[3] = height_axe
        ax2 = fig.add_axes(size_axe)
        ax2.set_xlabel('$k_h$')
        ax2.set_ylabel('transfers')
        title = ('Charney PE flux')
        ax2.set_title(title)
        ax2.hold(True)
        ax2.set_xscale('log')
        ax2.set_yscale('linear')

        if delta_t != 0.:
            for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
                transferCPE = dset_transfer2D_CPE[it]
                PiCPE = cumsum_inv(transferCPE)*self.oper.deltakh
                ax2.plot(khE, PiCPE, 'g', linewidth=1)

        transferCPE = dset_transfer2D_CPE[imin_plot:imax_plot].mean(0)
        PiCPE = cumsum_inv(transferCPE)*self.oper.deltakh

        ax2.plot(khE, PiCPE, 'm', linewidth=2)


