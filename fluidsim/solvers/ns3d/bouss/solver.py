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

from fluidfft.fft3d.operators import vector_product

from fluidsim.base.setofvariables import SetOfVariables

from ..strat.solver import InfoSolverNS3DStrat, Simul as SimulStrat


class InfoSolverNS3DBouss(InfoSolverNS3DStrat):
    def _init_root(self):

        super()._init_root()

        package = "fluidsim.solvers.ns3d.bouss"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "ns3d.bouss"


# classes = self.classes

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

    def tendencies_nonlin(self, state_spect=None, old=None):
        oper = self.oper
        ifft_as_arg = oper.ifft_as_arg
        ifft_as_arg_destroy = oper.ifft_as_arg_destroy
        fft_as_arg = oper.fft_as_arg

        if state_spect is None:
            spect_get_var = self.state.state_spect.get_var
        else:
            spect_get_var = state_spect.get_var

        vx_fft = spect_get_var("vx_fft")
        vy_fft = spect_get_var("vy_fft")
        vz_fft = spect_get_var("vz_fft")
        b_fft = spect_get_var("b_fft")

        omegax_fft, omegay_fft, omegaz_fft = oper.rotfft_from_vecfft(
            vx_fft, vy_fft, vz_fft
        )

        if self.params.f is not None:
            self._modif_omegafft_with_f(omegax_fft, omegay_fft, omegaz_fft)

        omegax = self.state.fields_tmp[3]
        omegay = self.state.fields_tmp[4]
        omegaz = self.state.fields_tmp[5]

        ifft_as_arg_destroy(omegax_fft, omegax)
        ifft_as_arg_destroy(omegay_fft, omegay)
        ifft_as_arg_destroy(omegaz_fft, omegaz)

        if state_spect is None:
            vx = self.state.state_phys.get_var("vx")
            vy = self.state.state_phys.get_var("vy")
            vz = self.state.state_phys.get_var("vz")
        else:
            vx = self.state.fields_tmp[0]
            vy = self.state.fields_tmp[1]
            vz = self.state.fields_tmp[2]
            ifft_as_arg(vx_fft, vx)
            ifft_as_arg(vy_fft, vy)
            ifft_as_arg(vz_fft, vz)

        fx, fy, fz = vector_product(vx, vy, vz, omegax, omegay, omegaz)

        if old is None:
            tendencies_fft = SetOfVariables(
                like=self.state.state_spect, info="tendencies_nonlin"
            )
        else:
            tendencies_fft = old

        fx_fft = tendencies_fft.get_var("vx_fft")
        fy_fft = tendencies_fft.get_var("vy_fft")
        fz_fft = tendencies_fft.get_var("vz_fft")

        fft_as_arg(fx, fx_fft)
        fft_as_arg(fy, fy_fft)
        fft_as_arg(fz, fz_fft)

        fz_fft += b_fft

        if state_spect is None:
            b = self.state.state_phys.get_var("b")
        else:
            b = self.state.fields_tmp[3]
            ifft_as_arg(b_fft, b)

        fb_fft = -oper.div_vb_fft_from_vb(vx, vy, vz, b)
        tendencies_fft.set_var("b_fft", fb_fft)

        if self.is_forcing_enabled:
            tendencies_fft += self.forcing.get_forcing()

        self.project_state_spect(tendencies_fft)
        self.oper.dealiasing(tendencies_fft)
        return tendencies_fft


if __name__ == "__main__":

    import numpy as np

    # import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    n = 64
    L = 10
    params.oper.nx = n * 2
    params.oper.ny = 32
    params.oper.nz = 8
    params.oper.Lx = L
    params.oper.Ly = L
    params.oper.Lz = L
    # params.oper.type_fft = 'fluidfft.fft3d.mpi_with_fftwmpi3d'
    # params.oper.type_fft = 'fluidfft.fft3d.with_pyfftw'
    # params.oper.type_fft = 'fluidfft.fft3d.with_cufft'

    # delta_x = params.oper.Lx / params.oper.nx
    params.nu_8 = 2e-6

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 10.0

    params.init_fields.type = "in_script"

    # params.forcing.enable = False
    # params.forcing.type = 'random'
    # 'Proportional'
    # params.forcing.type_normalize

    params.output.periods_print.print_stdout = 0.00000000001

    params.output.periods_save.phys_fields = 1.0
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

    variables = {
        k: 1e-6 * sim.oper.create_arrayX_random() for k in ("vx", "vy", "vz")
    }

    X, Y, Z = sim.oper.get_XYZ_loc()

    x0 = y0 = z0 = L / 2.0
    R2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
    r0 = 0.5
    b = -np.exp(-R2 / r0**2)
    variables["b"] = b

    sim.state.init_statephys_from(**variables)

    sim.state.statespect_from_statephys()
    sim.state.statephys_from_statespect()

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
# sim.output.phys_fields.plot()

# fld.show()
