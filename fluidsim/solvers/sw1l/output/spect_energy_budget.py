import numpy as np
import h5py

from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase,
    cumsum_inv,
    inner_prod,
)

from .normal_mode import NormalModeDecomposition, NormalModeDecompositionModified


class SpectralEnergyBudgetSW1LBase(SpectralEnergyBudgetBase):
    """Save and plot spectral energy budgets."""

    def __init__(self, output):

        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super().__init__(output)

    def _checksum_stdout(self, debug=False, **kwargs):
        if debug is True:
            for key, value in list(kwargs.items()):
                if mpi.nb_proc > 1:
                    value_g = mpi.comm.gather(value, root=0)
                else:
                    value_g = value

                if mpi.rank == 0:
                    print(
                        ("sum({0}) = {1:9.4e} ; sum(|{0}|) = {2:9.4e}").format(
                            key, np.sum(value_g), np.sum(np.abs(value_g))
                        )
                    )


class SpectralEnergyBudgetMSW1L(SpectralEnergyBudgetSW1LBase):
    def compute(self):
        """compute spectral energy budget the one time."""
        oper = self.sim.oper

        try:
            state_spect = self.sim.state.state_spect
            ux_fft = state_spect.get_var("ux_fft")
            uy_fft = state_spect.get_var("uy_fft")
            eta_fft = state_spect.get_var("eta_fft")
        except ValueError:
            state = self.sim.state
            ux_fft = state.get_var("ux_fft")
            uy_fft = state.get_var("uy_fft")
            eta_fft = state.get_var("eta_fft")

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        urx_fft, ury_fft = oper.vecfft_from_rotfft(rot_fft)
        del rot_fft
        urx = oper.ifft2(urx_fft)
        ury = oper.ifft2(ury_fft)

        q_fft, div_fft, a_fft = self.oper.qdafft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        udx_fft, udy_fft = oper.vecfft_from_divfft(div_fft)

        if self.params.f != 0:
            ugx_fft, ugy_fft, etag_fft = self.oper.uxuyetafft_from_qfft(q_fft)
            uax_fft, uay_fft, etaa_fft = self.oper.uxuyetafft_from_afft(a_fft)
        # velocity influenced by linear terms
        # u_infl_lin_x = udx_fft + uax_fft
        # u_infl_lin_y = udy_fft + uay_fft

        # compute flux of Charney PE
        Fq_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, q_fft)

        transferCPE_fft = inner_prod(q_fft, Fq_fft)
        del (q_fft, Fq_fft)

        transfer2D_CPE = self.spectrum2D_from_fft(transferCPE_fft)
        del transferCPE_fft

        #         print(
        # ('sum(transfer2D_CPE) = {0:9.4e} ; sum(abs(transfer2D_CPE)) = {1:9.4e}'
        # ).format(
        # np.sum(transfer2D_CPE),
        # np.sum(abs(transfer2D_CPE)))
        # )

        Feta_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, eta_fft)
        transferEA_fft = self.c2 * inner_prod(eta_fft, Feta_fft)
        del Feta_fft
        transfer2D_EA = self.spectrum2D_from_fft(transferEA_fft)
        del transferEA_fft

        #         print(
        # ('sum(transfer2D_EA) = {0:9.4e} ; sum(abs(transfer2D_EA)) = {1:9.4e}'
        # ).format(
        # np.sum(transfer2D_EA),
        # np.sum(abs(transfer2D_EA)))
        # )

        convA_fft = self.c2 * inner_prod(eta_fft, div_fft)
        convA2D = self.spectrum2D_from_fft(convA_fft)
        del convA_fft

        Fxrr_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, urx_fft)
        Fyrr_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, ury_fft)

        Fxrd_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, udx_fft)
        Fyrd_fft = self.fnonlinfft_from_uxuy_funcfft(urx, ury, udy_fft)

        transferErrr_fft = inner_prod(urx_fft, Fxrr_fft) + inner_prod(
            ury_fft, Fyrr_fft
        )
        transfer2D_Errr = self.spectrum2D_from_fft(transferErrr_fft)
        del transferErrr_fft
        #         print(
        # ('sum(transfer2D_Errr) = {0:9.4e} ; sum(abs(transfer2D_Errr)) = {1:9.4e}'
        # ).format(
        # np.sum(transfer2D_Errr),
        # np.sum(abs(transfer2D_Errr)))
        # )

        transferEdrd_fft = inner_prod(udx_fft, Fxrd_fft) + inner_prod(
            udy_fft, Fyrd_fft
        )
        transfer2D_Edrd = self.spectrum2D_from_fft(transferEdrd_fft)
        del transferEdrd_fft
        #         print(
        # ('sum(transfer2D_Edrd) = {0:9.4e} ; sum(abs(transfer2D_Edrd)) = {1:9.4e}'
        # ).format(
        # np.sum(transfer2D_Edrd),
        # np.sum(abs(transfer2D_Edrd)))
        # )

        Clfromqq = inner_prod(udx_fft, Fxrr_fft) + inner_prod(udy_fft, Fyrr_fft)
        transferEdrr_rrd_fft = (
            Clfromqq
            + inner_prod(urx_fft, Fxrd_fft)
            + inner_prod(ury_fft, Fyrd_fft)
        )
        Clfromqq = self.spectrum2D_from_fft(Clfromqq)
        transfer2D_Edrr_rrd = self.spectrum2D_from_fft(transferEdrr_rrd_fft)
        del transferEdrr_rrd_fft
        #         print(
        # ('sum(transfer2D_Edrr_rrd) = {0:9.4e} ; '
        # 'sum(abs(transfer2D_Edrr_rrd)) = {1:9.4e}'
        # ).format(
        # np.sum(transfer2D_Edrr_rrd),
        # np.sum(abs(transfer2D_Edrr_rrd)))
        # )

        transfer2D_EK = transfer2D_Errr + transfer2D_Edrd + transfer2D_Edrr_rrd
        dict_results = {
            "transfer2D_EK": transfer2D_EK,
            "transfer2D_Errr": transfer2D_Errr,
            "transfer2D_Edrd": transfer2D_Edrd,
            "Clfromqq": Clfromqq,
            "transfer2D_Edrr_rrd": transfer2D_Edrr_rrd,
            "transfer2D_EA": transfer2D_EA,
            "convA2D": convA2D,
            "transfer2D_CPE": transfer2D_CPE,
        }
        # self._checksum_stdout(
        #     EK_GGG=transfer2D_Errr,
        #     EK_GGA=transfer2D_Edrr_rrd,
        #     EK_AAG=transfer2D_Edrd,
        #     EA=transfer2D_EA,
        #     Etot=transfer2D_EK + transfer2D_EA,
        #     debug=False,
        # )
        return dict_results

    def _online_plot_saving(self, dict_results):

        transfer2D_CPE = dict_results["transfer2D_CPE"]
        transfer2D_EK = dict_results["transfer2D_EK"]
        transfer2D_EA = dict_results["transfer2D_EA"]
        convA2D = dict_results["convA2D"]
        khE = self.oper.khE
        PiCPE = cumsum_inv(transfer2D_CPE) * self.oper.deltak
        PiEK = cumsum_inv(transfer2D_EK) * self.oper.deltak
        PiEA = cumsum_inv(transfer2D_EA) * self.oper.deltak
        CCA = cumsum_inv(convA2D) * self.oper.deltak
        self.axe_a.plot(khE + khE[1], PiEK, "r")
        self.axe_a.plot(khE + khE[1], PiEA, "b")
        self.axe_a.plot(khE + khE[1], CCA, "y")
        self.axe_b.plot(khE + khE[1], PiCPE, "g")

    def plot(self, tmin=0, tmax=1000, delta_t=2):

        with h5py.File(self.path_file, "r") as h5file:

            dset_times = h5file["times"]
            dset_khE = h5file["khE"]
            khE = dset_khE[...]
            # khE = khE+khE[1]

            dset_transfer2D_EK = h5file["transfer2D_EK"]
            dset_transfer2D_Errr = h5file["transfer2D_Errr"]
            dset_transfer2D_Edrd = h5file["transfer2D_Edrd"]
            dset_transfer2D_Edrr_rrd = h5file["transfer2D_Edrr_rrd"]
            dset_transfer2D_EA = h5file["transfer2D_EA"]
            dset_convA2D = h5file["convA2D"]
            dset_transfer2D_CPE = h5file["transfer2D_CPE"]

            times = dset_times[...]
            # nt = len(times)

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            to_print = "plot(tmin={}, tmax={}, delta_t={:.2f})".format(
                tmin, tmax, delta_t
            )
            print(to_print)

            to_print = """plot fluxes 2D
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
            print(to_print)

            x_left_axe = 0.12
            z_bottom_axe = 0.36
            width_axe = 0.85
            height_axe = 0.57

            size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
            fig, ax1 = self.output.figure_axe(size_axe=size_axe)
            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel("transfers")
            ax1.set_title("energy flux\n" + self.output.summary_simul)
            ax1.set_xscale("log")
            ax1.set_yscale("linear")

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot, delta_i_plot):
                    transferEK = dset_transfer2D_EK[it]
                    transferEA = dset_transfer2D_EA[it]
                    PiEK = cumsum_inv(transferEK) * self.oper.deltak
                    PiEA = cumsum_inv(transferEA) * self.oper.deltak
                    PiE = PiEK + PiEA
                    ax1.plot(khE, PiE, "k", linewidth=1)

            transferEK = dset_transfer2D_EK[imin_plot:imax_plot].mean(0)
            transferEA = dset_transfer2D_EA[imin_plot:imax_plot].mean(0)
            PiEK = cumsum_inv(transferEK) * self.oper.deltak
            PiEA = cumsum_inv(transferEA) * self.oper.deltak
            PiE = PiEK + PiEA

            ax1.plot(khE, PiE, "k", linewidth=2)
            ax1.plot(khE, PiEK, "r", linewidth=2)
            ax1.plot(khE, PiEA, "b", linewidth=2)

            transferEdrr_rrd = dset_transfer2D_Edrr_rrd[imin_plot:imax_plot].mean(
                0
            )
            transferErrr = dset_transfer2D_Errr[imin_plot:imax_plot].mean(0)
            transferEdrd = dset_transfer2D_Edrd[imin_plot:imax_plot].mean(0)

            Pi_drr_rrd = cumsum_inv(transferEdrr_rrd) * self.oper.deltak
            Pi_rrr = cumsum_inv(transferErrr) * self.oper.deltak
            Pi_drd = cumsum_inv(transferEdrd) * self.oper.deltak

            ax1.plot(khE, Pi_drr_rrd, "m:", linewidth=1)
            ax1.plot(khE, Pi_rrr, "m--", linewidth=1)
            ax1.plot(khE, Pi_drd, "m-.", linewidth=1)

            convA2D = dset_convA2D[imin_plot:imax_plot].mean(0)
            CCA = cumsum_inv(convA2D) * self.oper.deltak

            ax1.plot(khE + khE[1], CCA, "y", linewidth=2)

            z_bottom_axe = 0.07
            height_axe = 0.17
            size_axe[1] = z_bottom_axe
            size_axe[3] = height_axe
            ax2 = fig.add_axes(size_axe)
            ax2.set_xlabel("$k_h$")
            ax2.set_ylabel("transfers")
            ax2.set_title("Charney PE flux")
            ax2.set_xscale("log")
            ax2.set_yscale("linear")

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    transferCPE = dset_transfer2D_CPE[it]
                    PiCPE = cumsum_inv(transferCPE) * self.oper.deltak
                    ax2.plot(khE, PiCPE, "g", linewidth=1)

            transferCPE = dset_transfer2D_CPE[imin_plot:imax_plot].mean(0)
            PiCPE = cumsum_inv(transferCPE) * self.oper.deltak

        ax2.plot(khE, PiCPE, "m", linewidth=2)


class SpectralEnergyBudgetSW1L(SpectralEnergyBudgetSW1LBase):
    def __init__(self, output, norm_mode=None):
        if norm_mode is None:
            self.norm_mode = NormalModeDecomposition(output)
        else:
            self.norm_mode = norm_mode

        super().__init__(output)

    def _transfer_keys_coeff(self):
        c2 = self.params.c2
        # TODO: Check again for possible bugs as found in SpectralEnergyBudgetSW1LModified
        keys = {
            "uuu": ["ux_fft", "ux", "px_ux"],  # Quad. K.E. transfer terms
            "uvu": ["ux_fft", "uy", "py_ux"],
            "vuv": ["uy_fft", "ux", "px_uy"],
            "vvv": ["uy_fft", "uy", "py_uy"],
            "eeu": ["px_eta_fft", "eta", "ux"],  # Quad. A.P.E. transfer terms
            "eev": ["py_eta_fft", "eta", "uy"],
            "uud": [
                "ux_fft",
                "ux",
                "div",
            ],  # NonQuad. K.E. - Quad K.E. transfer terms
            "vvd": ["uy_fft", "uy", "div"],
            "dee": ["div_fft", "eta", "eta"],
        }  # NonQuad. K.E. - Quad A.P.E. transfer terms

        coeff = {
            "uuu": -1.0,
            "uvu": -1.0,
            "vuv": -1.0,
            "vvv": -1.0,
            "eeu": 0.5 * c2,
            "eev": 0.5 * c2,
            "uud": -0.5,
            "vvd": -0.5,
            "dee": 0.25 * c2,
        }

        return keys, coeff

    def compute(self):
        """Compute spectral energy budget once for current time."""

        oper = self.oper
        get_var = self.sim.state.get_var
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        # eta_fft = get_var("eta_fft")
        c2 = self.params.c2

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")
        eta = self.sim.state.state_phys.get_var("eta")

        Mx = eta * ux
        My = eta * uy
        Mx_fft = oper.fft2(Mx)
        My_fft = oper.fft2(My)
        oper.dealiasing(Mx_fft, My_fft)
        del (Mx, My)

        self.norm_mode.compute()

        Tq_keys, Tq_coeff = self._transfer_keys_coeff()

        Tq_terms = dict.fromkeys(Tq_keys)
        for key in list(Tq_keys.keys()):
            triad_key_modes, Tq_terms[key] = self.norm_mode.triad_from_keyfftphys(
                *Tq_keys[key]
            )

        Tq_fft = dict.fromkeys(triad_key_modes[0], 0.0)
        n_modes = triad_key_modes[0].shape[0]
        for i in range(n_modes):  # GGG, GGA etc.
            k = triad_key_modes[0][i]
            for j in list(Tq_keys.keys()):  # uuu, uuv etc.
                Tq_fft[k] += np.real(Tq_coeff[j] * Tq_terms[j][i])

        del (Tq_keys, Tq_terms, Tq_coeff)

        # -------------------------
        # Quadratic exchange terms
        # -------------------------
        Cq_keys = {"ue": ["ux_fft", "px_eta_fft"], "ve": ["uy_fft", "py_eta_fft"]}
        Cq_terms = dict.fromkeys(Cq_keys)
        for key in list(Cq_keys.keys()):
            dyad_key_modes, Cq_terms[key] = self.norm_mode.dyad_from_keyfft(
                True, *Cq_keys[key]
            )

        Cq_coeff = {"ue": -c2, "ve": -c2}
        Cq_fft = dict.fromkeys(dyad_key_modes[0], 0.0)
        n_modes = dyad_key_modes[0].shape[0]
        for i in range(n_modes):  # GG, AG, aG, AA
            k = dyad_key_modes[0][i]
            for j in list(Cq_keys.keys()):  # ue, ve
                Cq_fft[k] += np.real(Cq_coeff[j] * Cq_terms[j][i])

        del (Cq_keys, Cq_terms, Cq_coeff)

        # -----------------------------------------------
        # Non-quadratic K.E. transfer and exchange terms
        # -----------------------------------------------
        inner_prod = lambda a_fft, b_fft: a_fft.conj() * b_fft
        inner_prod2 = lambda a_fft, b_fft: a_fft * b_fft.conj()
        triple_prod_conv = lambda ax_fft, ay_fft, bx_fft, by_fft: (
            inner_prod(ax_fft, self.fnonlinfft_from_uxuy_funcfft(ux, uy, bx_fft))
            + inner_prod(
                ay_fft, self.fnonlinfft_from_uxuy_funcfft(ux, uy, by_fft)
            )
        )
        triple_prod_conv2 = lambda ax_fft, ay_fft, bx_fft, by_fft: (
            inner_prod2(ax_fft, self.fnonlinfft_from_uxuy_funcfft(ux, uy, bx_fft))
            + inner_prod2(
                ay_fft, self.fnonlinfft_from_uxuy_funcfft(ux, uy, by_fft)
            )
        )

        u_udM = triple_prod_conv(ux_fft, uy_fft, Mx_fft, My_fft)
        M_udu = triple_prod_conv2(Mx_fft, My_fft, ux_fft, uy_fft)
        divM = oper.ifft2(oper.divfft_from_vecfft(Mx_fft, My_fft))

        ux_divM = oper.fft2(ux * divM)
        uy_divM = oper.fft2(uy * divM)
        oper.dealiasing(ux_divM, uy_divM)

        u_u_divM = inner_prod(ux_fft, ux_divM) + inner_prod(uy_fft, uy_divM)
        Tnq_fft = 0.5 * np.real(M_udu + u_udM - u_u_divM)
        del (M_udu, u_udM, divM, ux_divM, uy_divM, u_u_divM)

        # --------------------------------------
        # Enstrophy transfer terms
        # --------------------------------------
        Tens_fft = oper.K2 * Tq_fft["GGG"]

        Tq_GGG = self.spectrum2D_from_fft(Tq_fft["GGG"])
        Tq_AGG = self.spectrum2D_from_fft(Tq_fft["AGG"])
        Tq_GAAs = self.spectrum2D_from_fft(Tq_fft["GAAs"])
        Tq_GAAd = self.spectrum2D_from_fft(Tq_fft["GAAd"])
        Tq_AAA = self.spectrum2D_from_fft(Tq_fft["AAA"])
        Tnq = self.spectrum2D_from_fft(Tnq_fft)
        Tens = self.spectrum2D_from_fft(Tens_fft)
        Cq_GG = self.spectrum2D_from_fft(Cq_fft["GG"])
        Cq_AG = self.spectrum2D_from_fft(Cq_fft["AG"])
        Cq_aG = self.spectrum2D_from_fft(Cq_fft["aG"])
        Cq_AA = self.spectrum2D_from_fft(Cq_fft["AA"])

        # Tq_TOT = Tq_GGG + Tq_AGG + Tq_GAAs + Tq_GAAd + Tq_AAA
        # self._checksum_stdout(
        #   GGG=Tq_GGG, GGA=Tq_AGG, AAG=(Tq_GAAs+Tq_GAAd), AAA=Tq_AAA,
        #   TNQ=Tnq, TOTAL=Tq_TOT, debug=True)

        dict_results = {
            "Tq_GGG": Tq_GGG,
            "Tq_AGG": Tq_AGG,
            "Tq_GAAs": Tq_GAAs,
            "Tq_GAAd": Tq_GAAd,
            "Tq_AAA": Tq_AAA,
            "Tnq": Tnq,
            "Cq_GG": Cq_GG,
            "Cq_AG": Cq_AG,
            "Cq_aG": Cq_aG,
            "Cq_AA": Cq_AA,
            "Tens": Tens,
        }

        return dict_results

    def _online_plot_saving(self, dict_results):

        # Tens = dict_results["Tens"]
        Tq_GGG = dict_results["Tq_GGG"]
        Tq_AGG = dict_results["Tq_AGG"]
        Tq_GAAs = dict_results["Tq_GAAs"]
        Tq_GAAd = dict_results["Tq_GAAd"]
        Tq_AAA = dict_results["Tq_AAA"]
        Tq_tot = Tq_GGG + Tq_AGG + Tq_GAAs + Tq_GAAd + Tq_AAA

        Cq_GG = dict_results["Cq_GG"]
        Cq_AG = dict_results["Cq_AG"] + dict_results["Cq_aG"]
        Cq_AA = dict_results["Cq_AA"]
        Cq_tot = Cq_GG + Cq_AG + Cq_AA

        khE = self.oper.khE
        # Piens = cumsum_inv(Tens) * self.oper.deltak
        Pi_tot = cumsum_inv(Tq_tot) * self.oper.deltak
        Pi_GGG = cumsum_inv(Tq_GGG) * self.oper.deltak
        Pi_AGG = cumsum_inv(Tq_AGG) * self.oper.deltak
        Pi_GAAs = cumsum_inv(Tq_GAAs) * self.oper.deltak
        Pi_GAAd = cumsum_inv(Tq_GAAd) * self.oper.deltak
        Pi_AAA = cumsum_inv(Tq_AAA) * self.oper.deltak

        Cflux_tot = cumsum_inv(Cq_tot) * self.oper.deltak
        Cflux_GG = cumsum_inv(Cq_GG) * self.oper.deltak
        Cflux_AG = cumsum_inv(Cq_AG) * self.oper.deltak
        Cflux_AA = cumsum_inv(Cq_AA) * self.oper.deltak

        self.axe_a.plot(
            khE + khE[1], Pi_tot, "k", linewidth=2, label=r"$\Pi_{tot}$"
        )
        self.axe_a.plot(
            khE + khE[1], Pi_GGG, "g--", linewidth=1, label=r"$\Pi_{GGG}$"
        )
        # self.axe_a.plot(
        #     khE+khE[1], Piens, 'g:', linewidth=1, label=r'$\Pi_{ens}$')
        self.axe_a.plot(
            khE + khE[1], Pi_AGG, "m--", linewidth=1, label=r"$\Pi_{GGA}$"
        )
        self.axe_a.plot(
            khE + khE[1], Pi_GAAs, "r:", linewidth=1, label=r"$\Pi_{G\pm\pm}$"
        )
        self.axe_a.plot(
            khE + khE[1], Pi_GAAd, "b:", linewidth=1, label=r"$\Pi_{G\pm\mp}$"
        )
        self.axe_a.plot(
            khE + khE[1], Pi_AAA, "y--", linewidth=1, label=r"$\Pi_{AAA}$"
        )

        self.axe_b.plot(
            khE + khE[1], Cflux_tot, "k", linewidth=2, label=r"$\Sigma C_{tot}$"
        )
        self.axe_b.plot(
            khE + khE[1], Cflux_GG, "g:", linewidth=1, label=r"$\Sigma C_{GG}$"
        )
        self.axe_b.plot(
            khE + khE[1], Cflux_AG, "m--", linewidth=1, label=r"$\Sigma C_{GA}$"
        )
        self.axe_b.plot(
            khE + khE[1], Cflux_AA, "y--", linewidth=1, label=r"$\Sigma C_{AA}$"
        )

        if self.nb_saved_times == 2:
            self.axe_a.set_title(
                "Spectral Energy Budget\n" + self.output.summary_simul
            )
            self.axe_a.legend()
            self.axe_b.legend()
            self.axe_b.set_ylabel(r"$\Sigma C(k_h)$")

    def plot(self, tmin=0, tmax=1000, delta_t=2):

        with h5py.File(self.path_file, "r") as h5file:

            dset_times = h5file["times"]
            dset_khE = h5file["khE"]
            khE = dset_khE[...] + 0.1  # Offset for semilog plots
            times = dset_times[...]

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            to_print = "plot(tmin={}, tmax={}, delta_t={:.2f})".format(
                tmin, tmax, delta_t
            )
            print(to_print)

            to_print = "plot fluxes 2D" + (
                ", tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}"
                + ", imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}"
            ).format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
            print(to_print)

            # -------------------------
            #  Transfer terms
            # -------------------------
            x_left_axe = 0.12
            z_bottom_axe = 0.46
            width_axe = 0.85
            height_axe = 0.47
            size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
            fig1, ax1 = self.output.figure_axe(size_axe=size_axe)
            fig2, ax2 = self.output.figure_axe(size_axe=size_axe)

            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel(r"Transfer fluxes, $\Pi(k_h)$")

            z_bottom_axe = 0.07
            height_axe = 0.27
            size_axe[1] = z_bottom_axe
            size_axe[3] = height_axe
            ax11 = fig1.add_axes(size_axe)
            ax11.set_xlabel("$k_h$")
            ax11.set_ylabel("Transfer terms, $T(k_h)$")

            title = "Spectral Energy Budget\n" + self.output.summary_simul
            ax1.set_title(title)
            ax1.set_xscale("log")
            ax1.axhline()
            ax11.set_title(title)
            ax11.set_xscale("log")
            ax11.axhline()

            P = self.sim.params.forcing.forcing_rate
            norm = 1 if P == 0 else P
            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot, delta_i_plot):
                    transferEtot = 0.0
                    for key in ["GGG", "AGG", "GAAs", "GAAd", "AAA"]:
                        transferEtot += h5file["Tq_" + key][it]
                    PiEtot = cumsum_inv(transferEtot) * self.oper.deltak / norm
                    ax1.plot(khE, PiEtot, "k", linewidth=1)

            Tq_GGG = h5file["Tq_GGG"][imin_plot:imax_plot].mean(0) / norm
            Tq_AGG = h5file["Tq_AGG"][imin_plot:imax_plot].mean(0) / norm
            Tq_GAAs = h5file["Tq_GAAs"][imin_plot:imax_plot].mean(0) / norm
            Tq_GAAd = h5file["Tq_GAAd"][imin_plot:imax_plot].mean(0) / norm
            Tq_AAA = h5file["Tq_AAA"][imin_plot:imax_plot].mean(0) / norm
            Tnq = h5file["Tnq"][imin_plot:imax_plot].mean(0) / norm
            Tens = h5file["Tens"][imin_plot:imax_plot].mean(0) / norm
            Tq_tot = Tq_GGG + Tq_AGG + Tq_GAAs + Tq_GAAd + Tq_AAA

            Pi_GGG = cumsum_inv(Tq_GGG) * self.oper.deltak
            Pi_AGG = cumsum_inv(Tq_AGG) * self.oper.deltak
            Pi_GAAs = cumsum_inv(Tq_GAAs) * self.oper.deltak
            Pi_GAAd = cumsum_inv(Tq_GAAd) * self.oper.deltak
            Pi_AAA = cumsum_inv(Tq_AAA) * self.oper.deltak
            Pi_nq = cumsum_inv(Tnq) * self.oper.deltak
            Pi_ens = cumsum_inv(Tens) * self.oper.deltak
            Pi_tot = Pi_GGG + Pi_AGG + Pi_GAAs + Pi_GAAd + Pi_AAA

            ax1.plot(khE, Pi_GGG, "g--", linewidth=2, label=r"$\Pi_{GGG}$")
            ax1.plot(khE, Pi_AGG, "m--", linewidth=2, label=r"$\Pi_{GGA}$")
            # ax1.plot(khE, Pi_GAAs, 'r:', linewidth=2, label=r'$\Pi_{G\pm\pm}$')
            # ax1.plot(khE, Pi_GAAd, 'b:', linewidth=2, label=r'$\Pi_{G\pm\mp}$')
            ax1.plot(
                khE, Pi_GAAs + Pi_GAAd, "r", linewidth=2, label=r"$\Pi_{GAA}$"
            )
            ax1.plot(khE, Pi_AAA, "y--", linewidth=2, label=r"$\Pi_{AAA}$")
            ax1.plot(khE, Pi_nq, "k--", linewidth=1, label=r"$\Pi^{NQ}$")
            ax1.plot(khE, Pi_tot, "k", linewidth=3, label=r"$\Pi_{tot}$")
            ax1.legend()

            ax11.plot(khE, Tq_GGG, "g--", linewidth=2, label=r"$T_{GGG}$")
            ax11.plot(khE, Tq_AGG, "m--", linewidth=2, label=r"$T_{GGA}$")
            ax11.plot(khE, Tq_GAAs, "r:", linewidth=2, label=r"$T_{G\pm\pm}$")
            ax11.plot(khE, Tq_GAAd, "b:", linewidth=2, label=r"$T_{G\pm\mp}$")
            ax11.plot(khE, Tq_AAA, "y--", linewidth=2, label=r"$T_{AAA}$")
            ax11.plot(khE, Tnq, "k--", linewidth=2, label=r"$T^{NQ}$")
            ax11.plot(khE, Tq_tot, "k", linewidth=3, label=r"$T_{tot}$")
            ax11.legend()
            # -------------------------
            # Quadratic exchange terms
            # -------------------------
            ax2.set_xlabel(r"$k_h$")
            ax2.set_ylabel(r"Exchange fluxes, $\Sigma C$")
            ax2.set_xscale("log")
            ax2.axhline()

            # .. TODO: Normalize with energy??
            exchange_GG = h5file["Cq_GG"][imin_plot:imax_plot].mean(0)
            exchange_AG = h5file["Cq_AG"][imin_plot:imax_plot].mean(0) + h5file[
                "Cq_aG"
            ][imin_plot:imax_plot].mean(0)
            exchange_AA = h5file["Cq_AA"][imin_plot:imax_plot].mean(0)
            exchange_mean = exchange_GG + exchange_AG + exchange_AA

            Cflux_GG = cumsum_inv(exchange_GG) * self.oper.deltak
            Cflux_AG = cumsum_inv(exchange_AG) * self.oper.deltak
            Cflux_AA = cumsum_inv(exchange_AA) * self.oper.deltak
            Cflux_mean = cumsum_inv(exchange_mean) * self.oper.deltak

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot, delta_i_plot):
                    exchangetot = 0.0
                    for key in ["GG", "AG", "aG", "AA"]:
                        exchangetot += h5file["Cq_" + key][it]
                    Cfluxtot = cumsum_inv(exchangetot) * self.oper.deltak
                    ax2.plot(khE, Cfluxtot, "k", linewidth=1)

        ax2.plot(khE, Cflux_mean, "k", linewidth=4, label=r"$\Sigma C_{tot}$")
        ax2.plot(khE, Cflux_GG, "g:", linewidth=2, label=r"$\Sigma C_{GG}$")
        ax2.plot(khE, Cflux_AG, "m--", linewidth=2, label=r"$\Sigma C_{GA}$")
        ax2.plot(khE, Cflux_AA, "y--", linewidth=2, label=r"$\Sigma C_{AA}$")
        ax2.legend()

        ax22 = fig2.add_axes(size_axe)
        ax22.set_xscale("log")
        ax22.axhline()
        ax22.set_xlabel(r"$k_h$")
        ax22.set_ylabel(r"$\Pi_{ens}(k_h)$")

        ax22.plot(khE, Pi_ens, "g", linewidth=3, label=r"$\Pi_{ens}$")
        ax22.legend()

        fig1.canvas.draw()
        fig2.canvas.draw()


class SpectralEnergyBudgetSW1LModified(SpectralEnergyBudgetSW1L):
    def __init__(self, output):
        norm_mode = NormalModeDecompositionModified(output)
        super().__init__(output, norm_mode)

    def _transfer_keys_coeff(self):
        c2 = self.params.c2
        keys = {
            "uuu": ["ux_fft", "urx", "px_ux"],  # Quad. K.E. transfer terms
            "uvu": ["ux_fft", "ury", "py_ux"],
            "vuv": ["uy_fft", "urx", "px_uy"],
            "vvv": ["uy_fft", "ury", "py_uy"],
            "eeu": ["eta_fft", "urx", "px_eta"],  # Quad. A.P.E. transfer terms
            "eev": ["eta_fft", "ury", "py_eta"],
        }

        coeff = {
            "uuu": -1.0,
            "uvu": -1.0,
            "vuv": -1.0,
            "vvv": -1.0,
            "eeu": -c2,
            "eev": -c2,
        }

        return keys, coeff
