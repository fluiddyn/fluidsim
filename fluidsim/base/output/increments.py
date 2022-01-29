import h5py
import os
import numpy as np

from transonic import boost, Array
from fluiddyn.util import mpi

from .base import SpecificOutput

Ai = Array[np.int32, "1d"]
Af = Array[float, "2d"]


@boost
def strfunc_from_pdf(
    rxs: Ai, pdf: Af, values: Af, order: float, absolute: bool = False
):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    S_order = np.empty(rxs.shape)
    if absolute:
        values = abs(values)
    for irx in range(rxs.size):
        deltainc = abs(values[irx, 1] - values[irx, 0])
        S_order[irx] = deltainc * np.sum(pdf[irx] * values[irx] ** order)

    return S_order


class Increments(SpecificOutput):
    """A :class:`Increments` object handles the saving of pdf of
    increments.
    """

    _tag = "increments"
    _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "increments"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def __init__(self, output):
        params = output.sim.params
        self.nx = params.oper.nx

        self.nrx = min(self.nx // 16, 128)
        self.nrx = int(max(self.nrx, self.nx // 2))
        rmin = 1
        rmax = int(0.8 * self.nx)
        delta_logr = np.log(rmax / rmin) / (self.nrx - 1)
        logr = np.log(rmin) + delta_logr * np.arange(self.nrx)
        self.rxs = np.array(np.round(np.exp(logr)), dtype=np.int32)

        for ir in range(1, self.nrx):
            if self.rxs[ir - 1] >= self.rxs[ir]:
                self.rxs[ir] = self.rxs[ir - 1] + 1

        self.nbins = 400

        self.output = output
        self._init_path_files()

        if os.path.exists(self.path_file):
            if mpi.rank == 0:
                with h5py.File(self.path_file, "r") as h5file:
                    self.rxs = h5file["rxs"][...]
                    self.nbins = h5file["nbins"][...]
            if mpi.nb_proc > 1:
                self.rxs = mpi.comm.bcast(self.rxs)
                self.nbins = mpi.comm.bcast(self.nbins)

        self.nrx = self.rxs.size
        arrays_1st_time = {"rxs": self.rxs, "nbins": self.nbins}
        self._bins = np.arange(0.5, self.nbins, dtype=float) / self.nbins
        self.keys_vars_to_compute = list(output.sim.state.state_phys.keys)

        super().__init__(
            output,
            period_save=params.output.periods_save.increments,
            has_to_plot_saved=params.output.increments.HAS_TO_PLOT_SAVED,
            arrays_1st_time=arrays_1st_time,
        )

    def _init_online_plot(self):
        if mpi.rank == 0:
            self.fig, axe = self.output.figure_axe(numfig=5_000_000)
            self.axe = axe
            axe.set_xlabel(r"$\delta u_x (x)$")
            axe.set_ylabel("pdf")
            axe.set_title(
                r"pdf $\delta u_x (x)$" + "\n" + self.output.summary_simul
            )

    def _online_plot_saving(self, dict_results, key="rot"):
        """online plot on pdf"""
        pdf = dict_results["pdf_delta_" + key]
        pdf = pdf.reshape([self.nrx, self.nbins])
        valmin = dict_results["valmin_" + key]
        valmax = dict_results["valmax_" + key]

        for irx, rx in enumerate(self.rxs):
            values_inc = self.compute_values_inc(valmin[irx], valmax[irx])
            self.axe.plot(values_inc + irx, pdf[irx])

    def compute(self):
        """compute the values at one time."""
        dict_results = {}
        for key in self.keys_vars_to_compute:
            var = self.sim.state.get_var(key)

            pdf_var = np.empty([self.nrx, self.nbins])
            valmin = np.empty([self.nrx])
            valmax = np.empty([self.nrx])

            for irx, rx in enumerate(self.rxs):
                inc_var = self.oper.compute_increments_dim1(var, rx)
                (pdf_var[irx], bin_edges_var) = self.oper.pdf_normalized(
                    inc_var, self.nbins
                )
                valmin[irx] = bin_edges_var[0]
                valmax[irx] = bin_edges_var[self.nbins]

            dict_results["pdf_delta_" + key] = pdf_var.flatten()
            dict_results["valmin_" + key] = valmin
            dict_results["valmax_" + key] = valmax

        return dict_results

    def compute_values_inc(self, valmin, valmax):
        return valmin + (valmax - valmin) * self._bins

    def load(self):
        """load the saved pdf and return a dictionary."""
        with h5py.File(self.path_file, "r") as h5file:
            times = h5file["times"][...]

            list_base_keys = [
                "pdf_delta_",
                "valmin_",
                "valmax_",
                # 'struc_func_'
            ]

            dict_results = {"times": times}
            for key in self.keys_vars_to_compute:
                for base_key in list_base_keys:
                    result = h5file[base_key + key][...]
                    dict_results[base_key + key] = result

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

        to_print = (
            "plot structure functions\n"
            "tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}\n"
            "imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}"
        ).format(
            tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
        )
        print(to_print)

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

        S2_ux = self.strfunc_from_pdf(pdf_ux, values_inc_ux, 2)
        S2_uy = self.strfunc_from_pdf(pdf_uy, values_inc_uy, 2)

        S4_ux = self.strfunc_from_pdf(pdf_ux, values_inc_ux, 4)
        S4_uy = self.strfunc_from_pdf(pdf_uy, values_inc_uy, 4)

        flatnessL_bis = S4_ux / S2_ux**2
        flatnessT_bis = S4_uy / S2_uy**2

        ax2.plot(rxs, flatnessL_bis, "c--", linewidth=2)
        ax2.plot(rxs, flatnessT_bis, "m--", linewidth=2)

        cond = np.logical_and(rxs < 70 * deltax, rxs > 5 * deltax)
        ax2.plot(rxs[cond], 1e1 * rxs[cond] ** (-1), ":k", linewidth=2)

        ax2.plot(rxs, 3 * np.ones(rxs.shape), "k--", linewidth=0.5)

    def strfunc_from_pdf(self, pdf, values, order, absolute=False):
        r"""Following the identity:
        .. math::
            E(x^m) = \int_{-\inf}^{\inf} x^m p(x) dx

        In this case, replace x with increments,
        .. math::
            \delta u(r, x) = u(x+r) - u(x)

        Thus, for a every value of r the mean of increments are computed
        as follows:
        .. math::
            <(\delta u)^m>
                = \int_{-\inf}^{\inf} (\delta u)^m p(\delta u) d(\delta u)
                = d(\delta u) \Sigma (\delta u)^m p(\delta u)
        """
        return strfunc_from_pdf(self.rxs, pdf, values, float(order), absolute)

    def load_pdf_from_file(
        self, tmin=0, tmax=None, key_var="ux", irx_to_plot=None
    ):
        """Plot some pdf."""
        with h5py.File(self.path_file, "r") as h5file:
            times = h5file["times"][...]
            nt = len(times)

            if tmax is None:
                tmax = times.max()

            rxs = h5file["rxs"][...]

            oper = h5file["/info_simul/params/oper"]
            nx = oper.attrs["nx"]
            Lx = oper.attrs["Lx"]

            deltax = Lx / nx

            rxs = np.array(rxs, dtype=np.float64) * deltax

            # orders = h5file['orders'][...]

            # delta_t_save = np.mean(times[1:]-times[0:-1])
            # delta_t = delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            # tmin_plot = times[imin_plot]
            # tmax_plot = times[imax_plot]

            #         to_print = '''load pdf of the increments
            # tmin = {0:8.6g} ; tmax = {1:8.6g}
            # imin = {2:8d} ; imax = {3:8d}'''.format(
            # tmin_plot, tmax_plot,
            # imin_plot, imax_plot)
            #         print(to_print)

            if irx_to_plot is None:
                irx_to_plot = np.arange(rxs.size)

            nb_rx_to_plot = irx_to_plot.size

            # print 'irx_to_plot', irx_to_plot
            # print 'self.rxs[irx_to_plot]', self.rxs[irx_to_plot]

            pdf_timemean = np.zeros([nb_rx_to_plot, self.nbins])
            values_inc_timemean = np.zeros([nb_rx_to_plot, self.nbins])

            valmin_timemean = np.zeros([nb_rx_to_plot])
            valmax_timemean = np.zeros([nb_rx_to_plot])
            nb_timemean = 0

            for it in range(imin_plot, imax_plot + 1):
                nb_timemean += 1
                valmin = h5file["valmin_" + key_var][it]
                valmax = h5file["valmax_" + key_var][it]

                for irxp, irx in enumerate(irx_to_plot):
                    valmin_timemean[irxp] += valmin[irx]
                    valmax_timemean[irxp] += valmax[irx]

            valmin_timemean /= nb_timemean
            valmax_timemean /= nb_timemean

            for irxp, irx in enumerate(irx_to_plot):
                values_inc_timemean[irxp] = self.compute_values_inc(
                    valmin_timemean[irxp], valmax_timemean[irxp]
                )

            nt = 0
            for it in range(imin_plot, imax_plot + 1):
                nt += 1
                pdf_dvar2D = h5file["pdf_delta_" + key_var][it]
                pdf_dvar2D = pdf_dvar2D.reshape([self.nrx, self.nbins])
                valmin = h5file["valmin_" + key_var][it]
                valmax = h5file["valmax_" + key_var][it]

                for irxp, irx in enumerate(irx_to_plot):
                    pdf_dvar = pdf_dvar2D[irx]
                    values_inc = self.compute_values_inc(valmin[irx], valmax[irx])

                    pdf_timemean[irxp] += np.interp(
                        values_inc_timemean[irxp], values_inc, pdf_dvar
                    )

        pdf_timemean /= nt

        return pdf_timemean, values_inc_timemean, nb_rx_to_plot

    def plot_pdf(self, tmin=0, tmax=None, key_var="ux", order=0, nb_rx_to_plot=5):

        irx_to_plot = np.arange(
            0, self.rxs.size, self.rxs.size / nb_rx_to_plot
        ).astype(int)
        nb_rx_to_plot = irx_to_plot.size

        (
            pdf_timemean,
            values_inc_timemean,
            nb_rx_to_plot,
        ) = self.load_pdf_from_file(
            tmin=tmin, tmax=tmax, key_var=key_var, irx_to_plot=irx_to_plot
        )

        to_print = f"plot_pdf(tmin={tmin}, tmax={tmax})"
        print(to_print)

        fig, ax1 = self.output.figure_axe()
        ax1.set_title("pdf increments\n" + self.output.summary_simul)

        ax1.set_xscale("linear")
        ax1.set_yscale("linear")

        ax1.set_xlabel(key_var)
        ax1.set_ylabel(r"PDF x $\delta v^" + repr(order) + "$")

        colors = ["k", "y", "r", "b", "g", "m", "c"]

        for irxp, irx in enumerate(irx_to_plot):

            print("color = {}, rx = {}".format(colors[irxp], self.rxs[irx]))

            val_inc = values_inc_timemean[irxp]

            ax1.plot(
                val_inc,
                pdf_timemean[irxp] * val_inc**order,
                colors[irxp] + "x-",
                linewidth=1,
            )


# def iorder_from_order(self, order):
#     """Return the indice corresponding to one value of order."""
#     iorder = abs(self.orders-order).argmin()
#     if self.orders[iorder] != order:
#         raise ValueError(
#             'Order {0} has not been computed ?'.format(order)
#             )
#     return iorder
