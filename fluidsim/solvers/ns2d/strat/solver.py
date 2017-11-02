
"""NS2D solver (:mod:`fluidsim.solvers.ns2d.strat.solver`)
==========================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""
from __future__ import division

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.solvers.ns2d.solver import \
    InfoSolverNS2D, Simul as SimulNS2D

from .util_pythran import tendencies_nonlin_ns2dstrat


class InfoSolverNS2DStrat(InfoSolverNS2D):
    def _init_root(self):

        super(InfoSolverNS2DStrat, self)._init_root()

        package = 'fluidsim.solvers.ns2d.strat'
        self.module_name = package + '.solver'
        self.class_name = 'Simul'
        self.short_name = 'NS2D.strat'

        classes = self.classes

        classes.State.module_name = package + '.state'
        classes.State.class_name = 'StateNS2DStrat'

        classes.InitFields.module_name = package + '.init_fields'
        classes.InitFields.class_name = 'InitFieldsNS2DStrat'

        classes.Output.module_name = package + '.output'
        classes.Output.class_name = 'OutputStrat'

        classes.Forcing.module_name = 'fluidsim.solvers.ns2d' + '.forcing'
        classes.Forcing.class_name = 'ForcingNS2D'


class Simul(SimulNS2D):
    """Pseudo-spectral solver 2D incompressible Navier-Stokes equations.

    """
    InfoSolver = InfoSolverNS2DStrat

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulNS2D._complete_params_with_default(params)
        attribs = {'N': 1.}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft_as_arg = oper.fft_as_arg
        ifft_as_arg = oper.ifft_as_arg

        tendencies_fft = SetOfVariables(like=self.state.state_fft)
        Frot_fft = tendencies_fft.get_var('rot_fft')
        Fb_fft = tendencies_fft.get_var('b_fft')

        if state_fft is None:
            rot_fft = self.state.state_fft.get_var('rot_fft')
            b_fft = self.state.state_fft.get_var('b_fft')
            ux = self.state.state_phys.get_var('ux')
            uy = self.state.state_phys.get_var('uy')
        else:
            rot_fft = state_fft.get_var('rot_fft')
            b_fft = state_fft.get_var('b_fft')
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = self.state.field_tmp0
            uy = self.state.field_tmp1
            ifft_as_arg(ux_fft, ux)
            ifft_as_arg(uy_fft, uy)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_b_fft, py_b_fft = oper.gradfft_from_fft(b_fft)

        px_rot = self.state.field_tmp2
        py_rot = self.state.field_tmp3

        px_b = self.state.field_tmp4
        py_b = self.state.field_tmp5

        ifft_as_arg(px_rot_fft, px_rot)
        ifft_as_arg(py_rot_fft, py_rot)
        ifft_as_arg(px_b_fft, px_b)
        ifft_as_arg(py_b_fft, py_b)

        Frot, Fb = tendencies_nonlin_ns2dstrat(
            ux, uy, px_rot, py_rot, px_b, py_b, self.params.N)

        fft_as_arg(Fb, Fb_fft)
        fft_as_arg(Frot, Frot_fft)

        Frot_fft += px_b_fft

        oper.dealiasing(tendencies_fft)

        # T_rot = np.real(Frot_fft.conj()*rot_fft
        #                + Frot_fft*rot_fft.conj())/2.
        # print ('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot)))

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft

if __name__ == "__main__":

    from math import pi

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    params.oper.nx = nh = 32
    params.oper.ny = 32
    params.oper.Lx = params.oper.Ly = Lh = 2 * pi
    params.oper.coef_dealiasing = 0.5

    delta_x = Lh / nh

    params.nu_2 = 1.*10e-6
    # params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.N = 1.  # Brunt Vaisala frequency
    params.time_stepping.USE_CFL = True
    params.time_stepping.USE_T_END = True
    # params.time_stepping.deltat0 = 0.1
    # Period of time of the simulation
    params.time_stepping.t_end = 5.
    # params.time_stepping.it_end = 50

    params.init_fields.type = 'noise'

    params.FORCING = True
    params.forcing.type = 'tcrandom_anisotropic'
    params.forcing.nkmax_forcing = 5
    params.forcing.nkmin_forcing = 4
    params.forcing.tcrandom_anisotropic.angle = '45'

    # 'Proportional'
    # params.forcing.type_normalize

    params.output.sub_directory = 'tests'

    params.output.periods_print.print_stdout = 0.001

    params.output.periods_save.phys_fields = 1.
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 1.

    params.output.periods_plot.phys_fields = 5.

    params.output.ONLINE_PLOT_OK = True

    params.output.spectra.HAS_TO_PLOT_SAVED = True
    params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = False
    params.output.increments.HAS_TO_PLOT_SAVED = False

    params.output.phys_fields.field_to_plot = 'rot'

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
