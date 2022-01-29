"""Forcing (:mod:`fluidsim.solvers.ns2d.strat.forcing`)
=======================================================

.. autoclass:: ForcingNS2DStrat
   :members:

"""

from fluidsim.solvers.ns2d.forcing import ForcingNS2D


class ForcingNS2DStrat(ForcingNS2D):

    """Forcing class for the ns2d strat solver.

    .. inheritance-diagram:: ForcingNS2DStrat

    """

    def compute_coef_ab_normalize(
        self, constant_rate_of, key_forced, f_fft, var_fft, deltat
    ):

        if constant_rate_of not in ["energy", "energyK"]:
            raise ValueError

        if hasattr(self.forcing_maker, "oper_coarse"):
            oper = self.forcing_maker.oper_coarse
            state = self.forcing_maker.fstate_coarse
        else:
            oper = self.sim.oper
            state = self.sim.state

        sum_k = oper.sum_wavenumbers

        if key_forced == "rot_fft":
            vx_fft, vy_fft = oper.vecfft_from_rotfft(var_fft)
            fx_fft, fy_fft = oper.vecfft_from_rotfft(f_fft)
            a = deltat / 2 * sum_k(abs(fx_fft) ** 2 + abs(fy_fft) ** 2)
            b = sum_k(
                (vx_fft.conj() * fx_fft).real + (vy_fft.conj() * fy_fft).real
            )

            return a, b
        elif key_forced == "ap_fft":
            rot_fft, b_fft = state.compute_rotbfft_from_apfft(var_fft)
            frot_fft, fb_fft = state.compute_rotbfft_from_apfft(f_fft)

            vx_fft, vy_fft = oper.vecfft_from_rotfft(rot_fft)
            fx_fft, fy_fft = oper.vecfft_from_rotfft(frot_fft)

        else:
            raise ValueError

        a = deltat / 2 * sum_k(abs(fx_fft) ** 2 + abs(fy_fft) ** 2)
        b = sum_k((vx_fft.conj() * fx_fft).real + (vy_fft.conj() * fy_fft).real)

        if constant_rate_of == "energyK":
            return a, b

        N2 = self.sim.params.N**2
        a += deltat / 2 * sum_k(abs(fb_fft) ** 2) / N2
        b += sum_k((b_fft.conj() * fb_fft).real) / N2

        return a, b
