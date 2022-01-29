"""2D waves solver (:mod:`fluidsim.solvers.waves2d.solver`)
==================================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""

import numpy as np
from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral,
    InfoSolverPseudoSpectral,
)


class InfoSolverWaves2d(InfoSolverPseudoSpectral):
    def _init_root(self):

        super()._init_root()

        package = "fluidsim.solvers.waves2d"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "Waves2d"

        classes = self.classes

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateWaves"


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
    InfoSolver = InfoSolverWaves2d

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        SimulBasePseudoSpectral._complete_params_with_default(params)
        attribs = {"c2": 1.0, "f": 0}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_spect=None, old=None):

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        tendencies_fft[:] = 0.0

        return tendencies_fft

    def compute_freq_complex(self, key):
        assert key in ["f_fft", "g_fft"], "Unexpected key: " + key
        if key == "f_fft":
            omega = self.oper.create_arrayK(value=0)
        elif key == "g_fft":
            omega = 1.0j * np.sqrt(
                self.params.f**2 + self.params.c2 * self.oper.K2
            )
        return omega


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    nh = 32
    Lh = 2 * np.pi
    params.oper.nx = nh
    params.oper.ny = nh
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    # params.oper.type_fft = 'FFTWPY'

    delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = (
        2.0 * 10e-1 * params.forcing.forcing_rate ** (1 / 3) * delta_x**8
    )

    params.time_stepping.t_end = 1.0
    params.time_stepping.USE_CFL = False

    params.init_fields.type = "noise"

    # params.forcing.enable = True
    # params.forcing.type = 'tcrandom'
    # 'Proportional'
    # params.forcing.type_normalize

    # params.output.periods_print.print_stdout = 0.25

    params.output.periods_save.phys_fields = 0.5

    params.output.periods_plot.phys_fields = 0.0

    params.output.ONLINE_PLOT_OK = True

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
    # params.output.increments.HAS_TO_PLOT_SAVED = True

    params.output.phys_fields.field_to_plot = "f"

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
