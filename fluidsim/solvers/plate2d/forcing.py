"""
Plate2d forcing (:mod:`fluidsim.solvers.plate2d.forcing`)
===============================================================

"""

from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import \
    Proportional as ProportionalBase

from fluidsim.base.forcing.specific import \
    TimeCorrelatedRandomPseudoSpectral as TCRandomPS


class ForcingPlate2D(ForcingBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        classes = [Proportional, TimeCorrelatedRandomPseudoSpectral]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)


class TCRandomPSW(TCRandomPS):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(TCRandomPSW, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'w_fft'


class Proportional(ProportionalBase):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(Proportional, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'w_fft'


class TimeCorrelatedRandomPseudoSpectral(TCRandomPSW):
    def compute_forcingc_raw(self):
        Fw_fft = super(TimeCorrelatedRandomPseudoSpectral,
                       self).compute_forcingc_raw()

        return Fw_fft

    # def compute_forcing_2nd_degree_eq(self):
    #     """compute a forcing normalize with a 2nd degree eq."""

    #     w_fft = self.sim.state.state_fft.get_var('w_fft')
    #     vmax = self.sim.params.forcing.vmax
    #     n0 = w_fft.shape[0]
    #     n1 = w_fft.shape[1]
    #     zf = np.zeros((n0, n1), dtype=np.complex128)
    #     kfSI = 2*np.pi*2.5
    #     sk = 1.
    #     kfor = self.sim.oper.Lx*kfSI/(2*np.pi)
    #     rand = np.random.normal(loc=0.0, scale=1.0, size=(2*n0/3, 2*n1/3))
    #     for i0 in xrange(2*n0/3):
    #         for i1 in xrange(2*n1/3):
    #             zf[i0, i1] = ((1/(2*sk**2)) *
    #                           (np.exp(-np.sqrt(i0**2+i1**2)-kfor-1)**2) *
    #                           np.exp(2*np.pi*1j*rand[i0, i1]))

    #             # zIJ(1:(n3+1),1:(n3+1)) =
    #             # exp(-(sqrt(I.^2+J.^2)-kfor-1).^2/(2*sk^2))
    #             # .*exp(2*pi*1i*rand(size(I)));
    #     forcingW_fft = self.oper.nx_loc**2 * vmax / np.sum(zf) * zf
    #     self.forcingc_fft.set_var('w_fft', forcingW_fft)

    #     self.put_forcingc_in_forcing()
