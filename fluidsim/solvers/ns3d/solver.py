# -*- coding: utf-8 -*-

"""NS3D solver (:mod:`fluidsim.solvers.ns3d.solver`)
====================================================

.. autoclass:: Simul
   :members:
   :private-members:

.. todo::

   - 3D pseudo-spectral operator with parallel fft,
   - output and 3D plotting,

"""

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral, InfoSolverPseudoSpectral3D)


class InfoSolverNS3D(InfoSolverPseudoSpectral3D):
    def _init_root(self):

        super(InfoSolverNS3D, self)._init_root()

        package = 'fluidsim.solvers.ns3d'
        self.module_name = package + '.solver'
        self.class_name = 'Simul'
        self.short_name = 'ns3d'

        classes = self.classes

        classes.State.module_name = package + '.state'
        classes.State.class_name = 'StateNS3D'

        classes.InitFields.module_name = package + '.init_fields'
        classes.InitFields.class_name = 'InitFieldsNS3D'

        classes.Output.module_name = package + '.output'
        classes.Output.class_name = 'Output'

        del(classes.Forcing)
        classes._tag_children.remove('Forcing')
        # classes.Forcing.module_name = package + '.forcing'
        # classes.Forcing.class_name = 'ForcingNS3D'


class Simul(SimulBasePseudoSpectral):
    r"""Pseudo-spectral solver 3D incompressible Navier-Stokes equations.

    Not yet implemented!

    Notes
    -----

    .. |p| mathmacro:: \partial

    .. |vv| mathmacro:: \textbf{v}

    .. |kk| mathmacro:: \textbf{k}

    .. |ek| mathmacro:: \hat{\textbf{e}}_\textbf{k}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    This class is dedicated to solve with a pseudo-spectral method the
    incompressible Navier-Stokes equations (possibly with hyper-viscosity):

    .. math::
      \p_t \vv + \vv \cdot \bnabla \vv =
      - \bnabla p  - \nu_\alpha (-\Delta)^\alpha \vv,

    where :math:`\vv` is the non-divergent velocity (:math:`\bnabla
    \cdot \vv = 0`), :math:`p` is the pressure, :math:`\Delta` is the
    3D Laplacian operator.

    In Fourier space, these equations can be written as:

    .. math::
      \p_t \hat v = N(v) + L \hat v,

    where

    .. math::

      N(\vv) = -P_\perp \widehat{ \vv \cdot \bnabla \vv },

    .. math::

      L = - \nu_\alpha |\kk|^{2\alpha},

    with :math:`P_\perp = (1 - \ek \ek \cdot)` the operator projection
    on the plane perpendicular to the wave number :math:`\kk`.

    """
    InfoSolver = InfoSolverNS3D

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulBasePseudoSpectral._complete_params_with_default(params)

    def tendencies_nonlin(self, state_fft=None):
        oper = self.oper
        fft3d = oper.fft3d
        ifft3d = oper.ifft3d

        if state_fft is None:
            vx = self.state.state_phys.get_var('vx')
            vy = self.state.state_phys.get_var('vy')
            vz = self.state.state_phys.get_var('vz')
            vx_fft = vy_fft = vz_fft = None

        else:
            vx_fft = state_fft.get_var('vx_fft')
            vy_fft = state_fft.get_var('vy_fft')
            vz_fft = state_fft.get_var('vz_fft')
            vx = ifft3d(vx_fft)
            vy = ifft3d(vy_fft)
            vz = ifft3d(vz_fft)

        Fvx, Fvy, Fvz = oper.vgradv_from_v(
            vx, vy, vz, vx_fft, vy_fft, vz_fft)

        Fvx_fft = fft3d(Fvx)
        Fvy_fft = fft3d(Fvy)
        Fvz_fft = fft3d(Fvz)

        oper.project_perpk3d(Fvx_fft, Fvy_fft, Fvz_fft)

        tendencies_fft = SetOfVariables(
            like=self.state.state_fft,
            info='tendencies_nonlin')

        tendencies_fft.set_var('vx_fft', Fvx_fft)
        tendencies_fft.set_var('vy_fft', Fvy_fft)
        tendencies_fft.set_var('vz_fft', Fvz_fft)

        if self.params.FORCING:
            tendencies_fft += self.forcing.get_forcing()

        return tendencies_fft


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    n = 16
    L = 2*np.pi
    params.oper.nx = n
    params.oper.ny = n
    params.oper.nz = n
    params.oper.Lx = L
    params.oper.Ly = L
    params.oper.Lz = L

    delta_x = params.oper.Lx/params.oper.nx
    # params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.nu_8 = 2.*10e-1*delta_x**8

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 6.
    params.time_stepping.it_end = 2

    params.init_fields.type = 'dipole'

    params.FORCING = False
    # params.forcing.type = 'random'
    # 'Proportional'
    # params.forcing.type_normalize

    params.output.periods_print.print_stdout = 0.00000000001

    params.output.periods_save.phys_fields = 1.
    # params.output.periods_save.spectra = 0.5
    # params.output.periods_save.spatial_means = 0.05
    # params.output.periods_save.spect_energy_budg = 0.5

    # params.output.periods_plot.phys_fields = 0.0

    params.output.ONLINE_PLOT_OK = True

    # params.output.spectra.HAS_TO_PLOT_SAVED = True
    # params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    # params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True

    # params.output.phys_fields.field_to_plot = 'rot'

    sim = Simul(params)

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
