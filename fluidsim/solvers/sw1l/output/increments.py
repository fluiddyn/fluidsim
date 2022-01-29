import h5py
import numpy as np

from fluidsim.base.output.increments import Increments


class IncrementsSW1L(Increments):
    """A :class:`Increments` object handles the saving of pdf of
    increments.
    """

    def __init__(self, output):
        super().__init__(output)
        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

    def _online_plot_saving(self, dict_results, key="eta"):
        """online plot on pdf"""
        super()._online_plot_saving(dict_results, key=key)

    def compute(self):
        dict_results = super().compute()

        get_var = self.sim.state.get_var
        ux = get_var("ux")
        uy = get_var("uy")
        eta = get_var("eta")
        Jx = (1 + eta) * ux

        S_uL2JL = np.empty([self.nrx])
        S_uT2JL = np.empty([self.nrx])
        S_c2h2uL = np.empty([self.nrx])
        S_uT2uL = np.empty([self.nrx])

        for irx, rx in enumerate(self.rxs):
            inc_ux = self.oper.compute_increments_dim1(ux, rx)
            inc_uy = self.oper.compute_increments_dim1(uy, rx)
            inc_eta = self.oper.compute_increments_dim1(eta, rx)
            inc_Jx = self.oper.compute_increments_dim1(Jx, rx)
            inc_uy2 = inc_uy**2
            S_uL2JL[irx] = np.mean(inc_ux**2 * inc_Jx)
            S_uT2JL[irx] = np.mean(inc_uy2 * inc_Jx)
            S_c2h2uL[irx] = self.params.c2 * np.mean(inc_eta**2 * inc_ux)
            S_uT2uL[irx] = np.mean(inc_uy2 * inc_ux)

        dict_results["struc_func_uL2JL"] = S_uL2JL
        dict_results["struc_func_uT2JL"] = S_uT2JL
        dict_results["struc_func_c2h2uL"] = S_c2h2uL
        dict_results["struc_func_Kolmo"] = S_uL2JL + S_uT2JL + S_c2h2uL

        dict_results["struc_func_uT2uL"] = S_uT2uL

        return dict_results

    def plot(self, tmin=0, tmax=None, delta_t=2, order=2, yscale="log"):
        """Plot some structure functions."""
        with h5py.File(self.path_file, "r") as h5file:
            times = h5file["times"][...]
            # nt = len(times)

            if tmax is None:
                tmax = times.max()

            rxs = h5file["rxs"][...]

            oper = h5file["/info_simul/params/oper"]
            nx = oper.attrs["nx"]
            Lx = oper.attrs["Lx"]
            deltax = Lx / nx

        rxs = np.array(rxs, dtype=np.float64) * deltax

        # orders = h5file['orders'][...]
        # dset_struc_func_ux = h5file['struc_func_ux']
        # dset_struc_func_uy = h5file['struc_func_uy']

        if len(times) > 1:
            delta_t_save = np.mean(times[1:] - times[0:-1])
        else:
            delta_t_save = delta_t

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

        to_print = """plot structure functions
tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
            tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
        )
        print(to_print)

        pdf_eta, values_inc_eta, nb_rx_to_plot = self.load_pdf_from_file(
            tmin=tmin, tmax=tmax, key_var="eta"
        )

        pdf_ux, values_inc_ux, nb_rx_to_plot = self.load_pdf_from_file(
            tmin=tmin, tmax=tmax, key_var="ux"
        )

        pdf_uy, values_inc_uy, nb_rx_to_plot = self.load_pdf_from_file(
            tmin=tmin, tmax=tmax, key_var="uy"
        )

        # iorder = self.iorder_from_order(order)
        order = float(order)

        x_left_axe = 0.12
        z_bottom_axe = 0.56
        width_axe = 0.85
        height_axe = 0.37
        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("$r_x$")
        ax1.set_ylabel(r"$\langle \delta u^{" + f"{order}" + "} \\rangle$")
        ax1.set_title("struct. functions\n" + self.output.summary_simul)

        ax1.set_xscale("log")
        ax1.set_yscale(yscale)

        # So_eta = self.strfunc_from_pdf(pdf_eta, values_inc_eta, order)
        So_ux = self.strfunc_from_pdf(pdf_ux, values_inc_ux, order)
        So_uy = self.strfunc_from_pdf(pdf_uy, values_inc_uy, order)

        norm = rxs

        # ax1.set_ylabel('struct. functions, order = {0}'.format(order))
        # if delta_t != 0.:
        #     for it in range(imin_plot,imax_plot+1,delta_i_plot):
        #         struc_func_ux = dset_struc_func_ux[it]
        #         struc_func_ux = struc_func_ux.reshape(
        #             [self.norders, self.nrx])
        #         struc_func_uy = dset_struc_func_uy[it]
        #         struc_func_uy = struc_func_uy.reshape(
        #             [self.norders, self.nrx])

        #         ax1.plot(rxs, struc_func_ux[iorder], 'c', linewidth=1)
        #         ax1.plot(rxs, struc_func_uy[iorder], 'm', linewidth=1)

        # struc_func_ux = dset_struc_func_ux[imin_plot:imax_plot+1].mean(0)
        # struc_func_ux = struc_func_ux.reshape([self.norders, self.nrx])
        # struc_func_uy = dset_struc_func_uy[imin_plot:imax_plot+1].mean(0)
        # struc_func_uy = struc_func_uy.reshape([self.norders, self.nrx])

        # ax1.plot(rxs, struc_func_ux[iorder]/norm, 'c', linewidth=2)
        # ax1.plot(rxs, struc_func_uy[iorder]/norm, 'm', linewidth=2)

        ax1.plot(rxs, So_ux / norm, "c-.", linewidth=2)
        ax1.plot(rxs, So_uy / norm, "m-.", linewidth=2)
        if order % 2 == 1:
            ax1.plot(rxs, -So_ux / norm, "c:", linewidth=2)
            ax1.plot(rxs, -So_uy / norm, "m:", linewidth=2)

        # ax1.plot(rxs, abs(struc_func_ux[iorder])/abs(struc_func_uy[iorder]),
        #          'k', linewidth=1)

        ax1.plot(rxs, abs(So_ux) / abs(So_uy), "k", linewidth=1)

        # if self.orders[iorder]%2 == 1:
        #     ax1.plot(rxs, -struc_func_ux[iorder]/norm, '--b', linewidth=2)
        #     ax1.plot(rxs, -struc_func_uy[iorder]/norm, '--m', linewidth=2)

        cond = rxs < 6 * deltax
        ax1.plot(
            rxs[cond], 1.0e4 * rxs[cond] ** (order) / norm[cond], "k", linewidth=2
        )
        ax1.plot(rxs, rxs ** (order / 3) / norm, "--k", linewidth=2)

        ax1.plot(rxs, 1.0e0 * rxs ** (1) / norm, ":k", linewidth=2)

        z_bottom_axe = 0.09
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)

        ax2.set_xlabel("$r_x$")
        ax2.set_ylabel("flatness")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        # iorder4 = self.iorder_from_order(4)
        # iorder2 = self.iorder_from_order(2)

        # if delta_t != 0.:
        #     for it in range(imin_plot,imax_plot+1,delta_i_plot):
        #         struc_func_ux = dset_struc_func_ux[it]
        #         struc_func_ux = struc_func_ux.reshape(
        #             [self.norders, self.nrx])
        #         struc_func_uy = dset_struc_func_uy[it]
        #         struc_func_uy = struc_func_uy.reshape(
        #             [self.norders, self.nrx])

        #         flatnessL = struc_func_ux[iorder4]/struc_func_ux[iorder2]**2
        #         flatnessT = struc_func_uy[iorder4]/struc_func_uy[iorder2]**2

        #         ax2.plot(rxs, flatnessL, 'c', linewidth=1)
        #         ax2.plot(rxs, flatnessT, 'm', linewidth=1)

        # struc_func_ux = dset_struc_func_ux[imin_plot:imax_plot+1].mean(0)
        # struc_func_ux = struc_func_ux.reshape([self.norders, self.nrx])
        # struc_func_uy = dset_struc_func_uy[imin_plot:imax_plot+1].mean(0)
        # struc_func_uy = struc_func_uy.reshape([self.norders, self.nrx])

        # flatnessL = struc_func_ux[iorder4]/struc_func_ux[iorder2]**2
        # flatnessT = struc_func_uy[iorder4]/struc_func_uy[iorder2]**2
        # ax2.plot(rxs, flatnessL, 'c', linewidth=2)
        # ax2.plot(rxs, flatnessT, 'm', linewidth=2)

        S2_eta = self.strfunc_from_pdf(pdf_eta, values_inc_eta, 2)
        S2_ux = self.strfunc_from_pdf(pdf_ux, values_inc_ux, 2)
        S2_uy = self.strfunc_from_pdf(pdf_uy, values_inc_uy, 2)

        S4_eta = self.strfunc_from_pdf(pdf_eta, values_inc_eta, 4)
        S4_ux = self.strfunc_from_pdf(pdf_ux, values_inc_ux, 4)
        S4_uy = self.strfunc_from_pdf(pdf_uy, values_inc_uy, 4)

        flatnessL_bis = S4_ux / S2_ux**2
        flatnessT_bis = S4_uy / S2_uy**2
        flatness_eta = S4_eta / S2_eta**2

        ax2.plot(rxs, flatnessL_bis, "c--", linewidth=2)
        ax2.plot(rxs, flatnessT_bis, "m--", linewidth=2)
        ax2.plot(rxs, flatness_eta, "y--", linewidth=2)

        cond = np.logical_and(rxs < 70 * deltax, rxs > 5 * deltax)
        ax2.plot(rxs[cond], 1e1 * rxs[cond] ** (-1), ":k", linewidth=2)

        ax2.plot(rxs, 3 * np.ones(rxs.shape), "k--", linewidth=0.5)

    def plot_Kolmo(self, tmin=0, tmax=None):
        """Plot quantities appearing in the Kolmogorov law."""
        with h5py.File(self.path_file, "r") as h5file:
            times = h5file["times"][...]

            if tmax is None:
                tmax = times.max()

            rxs = h5file["rxs"][...]

            oper = h5file["/info_simul/params/oper"]
            nx = oper.attrs["nx"]
            Lx = oper.attrs["Lx"]
            deltax = Lx / nx

            rxs = np.array(rxs, dtype=np.float64) * deltax

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            to_print = f"plot(tmin={tmin}, tmax={tmax})"
            print(to_print)

            to_print = """plot structure functions
    tmin = {:8.6g} ; tmax = {:8.6g}
    imin = {:8d} ; imax = {:8d}""".format(
                tmin_plot, tmax_plot, imin_plot, imax_plot
            )
            print(to_print)

            # dset_struc_func_ux = h5file['struc_func_ux']
            # struc_func_ux = dset_struc_func_ux[imin_plot:imax_plot+1].mean(0)
            # struc_func_ux = struc_func_ux.reshape([self.norders, self.nrx])
            # order = 3
            # iorder = self.iorder_from_order(order)
            # S_ux3 = struc_func_ux[iorder]

            S_uL2JL = h5file["struc_func_uL2JL"][imin_plot : imax_plot + 1].mean(
                0
            )
            S_uT2JL = h5file["struc_func_uT2JL"][imin_plot : imax_plot + 1].mean(
                0
            )
            S_c2h2uL = h5file["struc_func_c2h2uL"][
                imin_plot : imax_plot + 1
            ].mean(0)
            S_Kolmo = h5file["struc_func_Kolmo"][imin_plot : imax_plot + 1].mean(
                0
            )
            # S_uT2uL = h5file["struc_func_uT2uL"][imin_plot : imax_plot + 1].mean(
            #     0
            # )

        S_Kolmo_theo = -4 * rxs

        x_left_axe = 0.12
        z_bottom_axe = 0.56
        width_axe = 0.85
        height_axe = 0.37
        size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel("$r_x$")
        ax1.set_title("struct. functions\n" + self.output.summary_simul)
        ax1.set_xscale("log")
        ax1.set_yscale("linear")

        ax1.set_ylabel("struct. functions")

        ax1.plot(rxs, S_Kolmo / S_Kolmo_theo, "k", linewidth=2)
        ax1.plot(rxs, (S_uL2JL + S_uT2JL) / S_Kolmo_theo, "r", linewidth=2)

        ax1.plot(rxs, S_c2h2uL / S_Kolmo_theo, "b", linewidth=2)

        ax1.plot(rxs, S_uL2JL / S_Kolmo_theo, "r--", linewidth=1)
        ax1.plot(rxs, S_uT2JL / S_Kolmo_theo, "r-.", linewidth=1)

        ax1.plot(
            rxs, (S_uL2JL + S_uT2JL + S_c2h2uL) / S_Kolmo_theo, "y", linewidth=1
        )

        cond = rxs < 6 * deltax
        ax1.plot(
            rxs[cond],
            1.0e0 * rxs[cond] ** 3 / S_Kolmo_theo[cond],
            "k",
            linewidth=2,
        )

        ax1.plot(rxs, np.ones(rxs.shape), "k:", linewidth=1)

        ax1.set_ylim([5e-2, 1.5])

        z_bottom_axe = 0.09
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)

        ax2.set_xlabel("$r_x$")
        ax2.set_ylabel("ratio S_ux3/S_uT2uL")
        ax2.set_xscale("log")
        ax2.set_yscale("linear")

        # ax2.plot(rxs, S_ux3/S_uT2uL, 'k', linewidth=2)

        ax2.plot(rxs, S_uL2JL / S_uT2JL, "k--", linewidth=2)

        ax2.plot(rxs, 3 * np.ones(rxs.shape), "k:", linewidth=1)

        ax2.set_ylim([2, 5])
