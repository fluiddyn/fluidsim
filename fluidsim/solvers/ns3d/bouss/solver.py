# -*- coding: utf-8 -*-

"""Boussinesq NS3D solver (:mod:`fluidsim.solvers.ns3d.bouss.solver`)
=====================================================================

.. autoclass:: InfoSolverNS3DBouss
   :members:
   :private-members:

.. autoclass:: Simul
   :members:
   :private-members:

"""
from __future__ import division

from fluidsim.base.setofvariables import SetOfVariables

from ..strat.solver import InfoSolverNS3DStrat, Simul as SimulStrat


class InfoSolverNS3DBouss(InfoSolverNS3DStrat):
    def _init_root(self):

        super(InfoSolverNS3DBouss, self)._init_root()

        package = 'fluidsim.solvers.ns3d.bouss'
        self.module_name = package + '.solver'
        self.class_name = 'Simul'
        self.short_name = 'ns3d.bouss'

        classes = self.classes

        # classes.State.module_name = package + '.state'
        # classes.State.class_name = 'StateNS3DStrat'

        # classes.InitFields.module_name = package + '.init_fields'
        # classes.InitFields.class_name = 'InitFieldsNS3D'

        # classes.Output.module_name = package + '.output'
        # classes.Output.class_name = 'Output'

        # classes.Forcing.module_name = package + '.forcing'
        # classes.Forcing.class_name = 'ForcingNS3D'


class Simul(SimulStrat):
    r"""Pseudo-spectral solver 3D incompressible Navier-Stokes equations.

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

      N(\vv) = -P_\perp \widehat{\bnabla \cdot \vv \vv},

    .. math::

      L = - \nu_\alpha |\kk|^{2\alpha},

    with :math:`P_\perp = (1 - \ek \ek \cdot)` the operator projection on the
    plane perpendicular to the wave number :math:`\kk`. Since the flow is
    incompressible (:math:`\kk \cdot \vv = 0`), the effect of the pressure term
    is taken into account with the operator :math:`P_\perp`.

    """
    InfoSolver = InfoSolverNS3DBouss

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        SimulStrat._complete_params_with_default(params)
        attribs = {'NO_SHEAR_MODES': False}
        params._set_attribs(attribs)

    def tendencies_nonlin(self, state_spect=None):
        oper = self.oper
        # fft3d = oper.fft3d
        ifft3d = oper.ifft3d

        if state_spect is None:
            vx = self.state.state_phys.get_var('vx')
            vy = self.state.state_phys.get_var('vy')
            vz = self.state.state_phys.get_var('vz')
            b = self.state.state_phys.get_var('b')
            vz_fft = self.state.state_spect.get_var('vz_fft')
            b_fft = self.state.state_spect.get_var('b_fft')
        else:
            vx_fft = state_spect.get_var('vx_fft')
            vy_fft = state_spect.get_var('vy_fft')
            vz_fft = state_spect.get_var('vz_fft')
            b_fft = state_spect.get_var('b_fft')
            vx = ifft3d(vx_fft)
            vy = ifft3d(vy_fft)
            vz = ifft3d(vz_fft)
            b = ifft3d(b_fft)

        Fvx_fft, Fvy_fft, Fvz_fft = oper.div_vv_fft_from_v(vx, vy, vz)        
        Fvx_fft, Fvy_fft, Fvz_fft = -Fvx_fft, -Fvy_fft, -Fvz_fft
        Fvz_fft += b_fft

        Fb_fft = -oper.div_vb_fft_from_vb(vx, vy, vz, b)
        
        oper.project_perpk3d(Fvx_fft, Fvy_fft, Fvz_fft)

        tendencies_fft = SetOfVariables(
            like=self.state.state_spect,
            info='tendencies_nonlin')

        tendencies_fft.set_var('vx_fft', Fvx_fft)
        tendencies_fft.set_var('vy_fft', Fvy_fft)
        tendencies_fft.set_var('vz_fft', Fvz_fft)
        tendencies_fft.set_var('b_fft', Fb_fft)

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
    params.oper.type_fft = 'fluidfft.fft3d.mpi_with_fftwmpi3d'
    # params.oper.type_fft = 'fluidfft.fft3d.with_fftw3d'
    # params.oper.type_fft = 'fluidfft.fft3d.with_cufft'

    delta_x = params.oper.Lx / params.oper.nx
    # params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.nu_8 = 2.*10e-1*delta_x**8

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 6.
    params.time_stepping.it_end = 2

    params.init_fields.type = 'in_script'

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

    # here we have to initialize the flow fields

    # we need to improve fluidfft for this.
    
    # sim.output.phys_fields.plot()
    # sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    # fld.show()
