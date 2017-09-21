
from .solver import InfoSolverNS2D as _InfoSolverNS2D, Simul as _Simul
from .util_pythran import compute_Frot
from fluidsim.base.setofvariables import SetOfVariables


class InfoSolverNS2D(_InfoSolverNS2D):
    def _init_root(self):
        super(InfoSolverNS2D, self)._init_root()

        self.module_name += '_oper_cython'

        self.classes.Operators.module_name = 'fluidsim.operators.operators'
        self.classes.Operators.class_name = 'OperatorsPseudoSpectral2D'


class Simul(_Simul):
    InfoSolver = InfoSolverNS2D

    def tendencies_nonlin(self, state_fft=None):
        r"""Compute the nonlinear tendencies.

        Parameters
        ----------

        state_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state, i.e. the vorticity, in Fourier
            space.  If `state_fft`, the variables vorticity and the
            velocity are computed from it, otherwise, they are taken
            from the global state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the vorticity.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The 2D Navier-Stockes equation can be written

        .. math:: \p_t \hat\zeta = \hat N(\zeta) - \sigma(k) \zeta,

        This function compute the nonlinear term ("tendencies")
        :math:`\hat N(\zeta) = - \mathbf{u}\cdot \mathbf{\nabla}
        \zeta`.

        """
        # the operator and the fast Fourier transform
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        # get or compute rot_fft, ux and uy
        if state_fft is None:
            rot_fft = self.state.state_fft.get_var('rot_fft')
            ux = self.state.state_phys.get_var('ux')
            uy = self.state.state_phys.get_var('uy')
        else:
            rot_fft = state_fft.get_var('rot_fft')
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = ifft2(ux_fft)
            uy = ifft2(uy_fft)

        # "px" like $\partial_x$
        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = ifft2(px_rot_fft)
        py_rot = ifft2(py_rot_fft)

        Frot = compute_Frot(ux, uy, px_rot, py_rot, self.params.beta)

        Frot_fft = fft2(Frot)
        oper.dealiasing(Frot_fft)

        # T_rot = np.real(Frot_fft.conj()*rot_fft
        #                + Frot_fft*rot_fft.conj())/2.
        # print ('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot)))

        tendencies_fft = SetOfVariables(like=self.state.state_fft)
        tendencies_fft.set_var('rot_fft', Frot_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":
    import numpy as np

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 16
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 5.

    params.init_fields.type = 'noise'

    params.output.periods_plot.phys_fields = 0.

    params.output.periods_print.print_stdout = 0.25
    params.output.periods_save.phys_fields = 2.
    params.output.periods_save.spectra = 1.

    sim = Simul(params)

    sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
