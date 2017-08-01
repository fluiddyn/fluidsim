"""NS2D solver (:mod:`fluidsim.solvers.ns2d.solver`)
====================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""
from __future__ import division

from fluidsim.base.setofvariables import SetOfVariables

# from fluidsim.base.solvers.pseudo_spect import (
#     SimulBasePseudoSpectral, InfoSolverPseudoSpectral)

from fluidsim.solvers.ns2d.solver import \
    InfoSolverNS2D, Simul as SimulNS2D


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

        # classes.Forcing.module_name = package + '.forcing'
        # classes.Forcing.class_name = 'ForcingNS2D'


class Simul(SimulNS2D):
    """Pseudo-spectral solver 2D incompressible Navier-Stokes equations.

    """
    InfoSolver = InfoSolverNS2DStrat

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulNS2D._complete_params_with_default(params)
        attribs = {'beta': 0., 'N': 1.}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        if state_fft is None:
            rot_fft = self.state.state_fft.get_var('rot_fft')
            b_fft = self.state.state_fft.get_var('b_fft')
            ux = self.state.state_phys.get_var('ux')
            uy = self.state.state_phys.get_var('uy')
        else:
            rot_fft = state_fft.get_var('rot_fft')
            b_fft = state_fft.get_var('b_fft')
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = ifft2(ux_fft)
            uy = ifft2(uy_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_b_fft, py_b_fft = oper.gradfft_from_fft(b_fft)
        px_rot = ifft2(px_rot_fft)
        py_rot = ifft2(py_rot_fft)
        px_b = ifft2(px_b_fft)
        py_b = ifft2(py_b_fft)

        if self.params.beta == 0:
            Frot = -ux*px_rot - uy*py_rot
            Fb = -ux*px_b - uy*py_b - self.params.N**2*uy
        else:
            Frot = -ux*px_rot - uy*(py_rot + self.params.beta)

        Fb_fft = fft2(Fb)
        Frot_fft_old = fft2(Frot)
        Frot_fft = Frot_fft_old + px_b_fft
        oper.dealiasing(Frot_fft)
        oper.dealiasing(Fb_fft)

        # T_rot = np.real(Frot_fft.conj()*rot_fft
        #                + Frot_fft*rot_fft.conj())/2.
        # print ('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot)))

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        tendencies_fft.set_var('rot_fft', Frot_fft)
        tendencies_fft.set_var('b_fft', Fb_fft)

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
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.N = 1.  # Brunt Vaisala frequency
    params.time_stepping.USE_CFL = False
    params.time_stepping.deltat0 = 0.1
    params.time_stepping.t_end = 20.  # Period of time of the simulation

    params.init_fields.type = 'dipole'

    params.FORCING = False
    params.forcing.type = 'random'
    # 'Proportional'
    # params.forcing.type_normalize

    params.output.sub_directory = 'tests'

    params.output.periods_print.print_stdout = 0.0002

    params.output.periods_save.phys_fields = 10.
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 1.

    params.output.periods_plot.phys_fields = 4.

    params.output.ONLINE_PLOT_OK = True

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    params.output.phys_fields.field_to_plot = 'rot'

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
