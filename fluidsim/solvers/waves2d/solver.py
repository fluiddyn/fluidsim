"""2D waves solver (:mod:`fluidsim.solvers.waves2d.solver`)
==================================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral)


info_solver = InfoSolverPseudoSpectral()

package = 'fluidsim.solvers.waves2d'
info_solver.module_name = package + '.solver'
info_solver.class_name = 'Simul'
info_solver.short_name = 'Waves2d'

classes = info_solver.classes

# classes.State.module_name = package + '.state'
# classes.State.class_name = 'StateWaves'

# classes.InitFields.module_name = package + '.init_fields'
# classes.InitFields.class_name = 'InitFieldsWaves'

# classes.Output.module_name = package + '.output'
# classes.Output.class_name = 'Output'

# classes.Forcing.module_name = package + '.forcing'
# classes.Forcing.class_name = 'ForcingNS2D'


info_solver.complete_with_classes()


class Simul(SimulBasePseudoSpectral):
    r"""Pseudo-spectral solver for equations of 2D waves.

    Notes
    -----

    .. |p| mathmacro:: \partial

    This class is dedicated to solve wave 2D equations:

    .. math::
       \p_t \hat f = \hat g - \gamma_f \hat f,

       \p_t \hat g = -\Omega^2 \hat f - \gamma_g \hat g,

    This purely linear wave equation can alternatively be written as
    as :math:`\p_t X = M X`, with

    .. math::

       X = \begin{pmatrix} \hat f \\ \hat g \end{pmatrix}
       \ \ \text{and}\ \
       M = \begin{pmatrix} -\gamma_f & 1 \\
       -\Omega^2 & -\gamma_g \end{pmatrix},

    where the three coefficients usually depend on the wavenumber.
    The eigenvalues are :math:`\sigma_\pm = - \bar \gamma \pm i \tilde
    \Omega`, where :math:`\bar \gamma = (\gamma_f + \gamma_g)/2` and

    .. math::

       \tilde \Omega = \Omega \sqrt{1 +
       \frac{1}{\Omega^2}(\gamma_f\gamma_g - \bar \gamma^2)}.

    The (not normalized) eigenvectors can be expressed as

    .. math::

       V_\pm = \begin{pmatrix} 1 \\ \sigma_\pm + \gamma_f \end{pmatrix}.

    The state can be represented by a vector :math:`A` that verifies
    :math:`X = V A`, where :math:`V` is the base matrix

    .. math::

       V = \begin{pmatrix} 1 & 1 \\
       \sigma_+ + \gamma_f & \sigma_- + \gamma_f \end{pmatrix}.

    The inverse base matrix is given by

    .. math::

       V^{-1} = \frac{i}{2\tilde \Omega}
       \begin{pmatrix}
       \sigma_- + \gamma_f & -1 \\
       -\sigma_+ - \gamma_f &  1 \end{pmatrix}.


    """

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {'c2': 1., 'f': 0}
        params._set_attribs(attribs)

    def __init__(self, params):
        super(Simul, self).__init__(params, info_solver)

    def tendencies_nonlin(self, state_fft=None):

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        tendencies_fft[:] = 0.

        return tendencies_fft

    def compute_freq_complex(self, key):
        if key == 'f_fft':
            omega = self.oper.constant_arrayK(value=0)
        elif key == 'g_fft':
            omega = 1.j*np.sqrt(self.params.f**2 +
                                self.params.c2*self.oper.K2)
        return omega


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

    params.init_fields.type = 'noise'

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
