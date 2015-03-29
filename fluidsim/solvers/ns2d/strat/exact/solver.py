"""Stratified NS2D solver (:mod:`fluidsim.solvers.ns2d.strat.solver`)
===========================================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""

from fluidsim.operators.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral)


info_solver = InfoSolverPseudoSpectral()

package = 'fluidsim.solvers.ns2d'
info_solver.module_name = package + '.solver'
info_solver.class_name = 'Simul'
info_solver.short_name = 'NS2D'

classes = info_solver.classes

classes.State.module_name = package + '.state'
classes.State.class_name = 'StateNS2D'

classes.InitFields.module_name = package + '.init_fields'
classes.InitFields.class_name = 'InitFieldsNS2D'

classes.Output.module_name = package + '.output'
classes.Output.class_name = 'Output'

# classes.Forcing.module_name = package + '.forcing'
# classes.Forcing.class_name = 'ForcingNS2D'


info_solver.complete_with_classes()


class Simul(SimulBasePseudoSpectral):
    r"""Pseudo-spectral solver strat. 2D incompressible Navier-Stokes equations.

    


    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {'N': 1.}
        params.set_attribs(attribs)

    def __init__(self, params):
        super(Simul, self).__init__(params, info_solver)

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft2 = oper.fft2
        ifft2 = oper.ifft2

        if state_fft is None:
            rot_fft = self.state.state_fft['rot_fft']
            b_fft = self.state.state_fft['b_fft']
            ux = self.state.state_phys['ux']
            uz = self.state.state_phys['uz']
        else:
            rot_fft = state_fft['rot_fft']
            b_fft = state_fft['b_fft']
            ux_fft, uz_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = ifft2(ux_fft)
            uz = ifft2(uz_fft)

        px_rot_fft, pz_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = ifft2(px_rot_fft)
        pz_rot = ifft2(pz_rot_fft)

        px_b_fft = oper.pxffft_ftt(b_fft)

        Frot = - ux*px_rot - uz*pz_rot

        Frot_fft = fft2(Frot) + px_b_fft
        oper.dealiasing(Frot_fft)

        Fb_fft = self.params.N2*uz_fft

        # T_rot = np.real(Frot_fft.conj()*rot_fft
        #                + Frot_fft*rot_fft.conj())/2.
        # print ('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}'
        #       ).format(self.oper.sum_wavenumbers(T_rot),
        #                self.oper.sum_wavenumbers(abs(T_rot)))

        tendencies_fft = SetOfVariables(
            like_this_sov=self.state.state_fft,
            name_type_variables='tendencies_nonlin')

        tendencies_fft['rot_fft'] = Frot_fft
        tendencies_fft['b_fft'] = Fb_fft

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = fld.simul.create_params(info_solver)

    params.short_name_type_run = 'test'

    nh = 32
    Lh = 2*np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    # params.oper.type_fft = 'FFTWPY'

    delta_x = params.oper.Lx/params.oper.nx
    params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8

    params.time_stepping.t_end = 1.

    params.init_fields.type_flow_init = 'NOISE'

    params.FORCING = True
    params.forcing.type = 'Random'
    # 'Proportional'
    # params.forcing.type_normalize

    # params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 0.5
    params.output.periods_save.increments = 0.5

    params.output.periods_plot.phys_fields = 0.0

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
