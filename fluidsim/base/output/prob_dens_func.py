
import h5py
import numpy as np

from fluidsim.base.output.base import SpecificOutput


class ProbaDensityFunc(SpecificOutput):
    """Handle the saving and plotting of pdf of the turbulent kinetic energy.
    """
    _tag = 'pdf'
    _name_file = _tag + '.h5'

    @staticmethod
    def _complete_params_with_default(params):
        tag = 'pdf'

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag,
                                 attribs={'HAS_TO_PLOT_SAVED': False})

    def __init__(self, output):
        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f
        self.nx = params.oper.nx

        super(ProbaDensityFunc, self).__init__(
            output,
            period_save=params.output.periods_save.pdf,
            has_to_plot_saved=params.output.pdf.HAS_TO_PLOT_SAVED)

    def init_online_plot(self):
        self.fig, axe = self.output.figure_axe(numfig=5000000)
        self.axe = axe
        axe.set_xlabel('$\eta$')
        axe.set_ylabel('pdf')
        title = ('pdf $\eta$, solver ' + self.output.name_solver +
                 ', nh = {0:5d}'.format(self.nx) +
                 ', c = {0:.4g}, f = {1:.4g}'.format(np.sqrt(self.c2), self.f))
        axe.set_title(title)
        axe.hold(True)

    def _online_plot(self, dico_pdf):
        """online plot on pdf"""
        pdf_eta = dico_pdf['pdf_eta']
        bin_edges_eta = dico_pdf['bin_edges_eta']
        self.axe.plot(bin_edges_eta[:-1], pdf_eta, 'k')

    def compute(self):
        """compute the values at one time."""
        eta = self.sim.state.state_phys.get_var('eta')
        pdf_eta, bin_edges_eta = self.oper.pdf_normalized(eta)

        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')

        uxm = ux.mean()
        uym = uy.mean()

        u_norme = np.sqrt((ux - uxm)**2 + (uy - uym)**2)
        pdf_u, bin_edges_u = self.oper.pdf_normalized(u_norme)

        dico_pdf = {'pdf_eta': pdf_eta,
                    'bin_edges_eta': bin_edges_eta,
                    'pdf_u': pdf_u,
                    'bin_edges_u': bin_edges_u}
        return dico_pdf

    def load(self):
        """load the saved pdf and return a dictionary."""
        f = h5py.File(self.path_file, 'r')
        # dset_times = f['times']
        # times = dset_times[...]
        # nt = len(times)

        dset_pdf_eta = f['pdf_eta']
        dset_bin_edges_eta = f['bin_edges_eta']

        pdf_eta = dset_pdf_eta[...]
        bin_edges_eta = dset_bin_edges_eta[...]

        dset_pdf_u = f['pdf_u']
        dset_bin_edges_u = f['bin_edges_u']

        pdf_u = dset_pdf_u[...]
        bin_edges_u = dset_bin_edges_u[...]

        dico_pdf = {'pdf_eta': pdf_eta,
                    'bin_edges_eta': bin_edges_eta,
                    'pdf_u': pdf_u,
                    'bin_edges_u': bin_edges_u}
        return dico_pdf

    def plot(self, tmin=0, tmax=1000, delta_t=2):
        """Plot some pdf."""
        f = h5py.File(self.path_file, 'r')
        dset_times = f['times']
        times = dset_times[...]
        # nt = len(times)

        dset_pdf_eta = f['pdf_eta']
        dset_bin_edges_eta = f['bin_edges_eta']
        dset_pdf_u = f['pdf_u']
        dset_bin_edges_u = f['bin_edges_u']

        delta_t_save = np.mean(times[1:]-times[0:-1])
        delta_i_plot = int(np.round(delta_t/delta_t_save))
        if delta_i_plot == 0:
            delta_i_plot = 1
        delta_t = delta_i_plot*delta_t_save

        imin_plot = np.argmin(abs(times-tmin))
        imax_plot = np.argmin(abs(times-tmax))

        tmin_plot = times[imin_plot]
        tmax_plot = times[imax_plot]

        to_print = 'plot(tmin={0}, tmax={1}, delta_t={2:.2f})'.format(
            tmin, tmax, delta_t)
        print(to_print)

        to_print = ('plot pdf eta\n'
                    'tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}'
                    'imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}').format(
                        tmin_plot, tmax_plot, delta_t,
                        imin_plot, imax_plot, delta_i_plot)
        print(to_print)

        x_left_axe = 0.12
        z_bottom_axe = 0.56
        width_axe = 0.85
        height_axe = 0.37
        size_axe = [x_left_axe, z_bottom_axe,
                    width_axe, height_axe]
        fig, ax1 = self.output.figure_axe(size_axe=size_axe)
        ax1.set_xlabel('$\eta$')
        ax1.set_ylabel('PDF')
        ax1.set_title('PDF, solver ' + self.output.name_solver +
                      ', nh = {0:5d}'.format(self.nx) +
                      ', c = {0:.4g}, f = {1:.4g}'.format(
                          np.sqrt(self.c2), self.f))
        ax1.hold(True)
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')

        for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
            pdf_eta = dset_pdf_eta[it]
            bin_edges_eta = dset_bin_edges_eta[it]

            bin_edges_eta = (bin_edges_eta[:-1]+bin_edges_eta[1:])/2
            ax1.plot(bin_edges_eta, pdf_eta, 'c', linewidth=1)

        z_bottom_axe = 0.09
        size_axe[1] = z_bottom_axe
        ax2 = fig.add_axes(size_axe)

        ax2.set_xlabel('$ |\\mathbf{u}-\\langle \\mathbf{u} \\rangle | $')
        ax2.set_ylabel('PDF')
        ax2.hold(True)
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')

        for it in xrange(imin_plot, imax_plot+1, delta_i_plot):
            pdf_u = dset_pdf_u[it]
            bin_edges_u = dset_bin_edges_u[it]

            bin_edges_u = (bin_edges_u[:-1]+bin_edges_u[1:])/2
            ax2.plot(bin_edges_u, pdf_u, 'r', linewidth=1)
