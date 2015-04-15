

from fluidsim.solvers.sw1l.init_fields import InitFieldsSW1L


class InitFieldsSW1LModified(InitFieldsSW1L):
    """Init """


    def fill_state_from_uxuyfft(self, ux_fft, uy_fft):
        sim = self.sim
        oper = sim.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self.etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        state_fft = sim.state.state_fft
        state_fft.set_var('ux_fft', ux_fft)
        state_fft.set_var('uy_fft', uy_fft)
        state_fft.set_var('eta_fft', eta_fft)

        state_phys = sim.state.state_phys
        state_phys.set_var('ux', ux)
        state_phys.set_var('uy', uy)
        state_phys.set_var('eta', eta)
