# -*- coding: utf-8 -*-

"""NS3D solver (:mod:`fluidsim.solvers.ns3d.solver`)
====================================================

.. autoclass:: Simul
   :members:
   :private-members:


"""

import sys

import numpy as np

from fluiddyn.util.mpi import rank

from fluidfft.fft3d.operators import vector_product

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.operators.operators3d import dealiasing_variable

from fluidsim.base.solvers.pseudo_spect import (
    SimulBasePseudoSpectral,
    InfoSolverPseudoSpectral3D,
)


class InfoSolverNS3D(InfoSolverPseudoSpectral3D):
    def _init_root(self):

        super()._init_root()

        package = "fluidsim.solvers.ns3d"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "ns3d"

        classes = self.classes

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateNS3D"

        classes.TimeStepping.module_name = package + ".time_stepping"
        classes.TimeStepping.class_name = "TimeSteppingPseudoSpectralNS3D"

        classes.InitFields.module_name = package + ".init_fields"
        classes.InitFields.class_name = "InitFieldsNS3D"

        classes.Output.module_name = package + ".output"
        classes.Output.class_name = "Output"

        classes.Forcing.module_name = package + ".forcing"
        classes.Forcing.class_name = "ForcingNS3D"


class Simul(SimulBasePseudoSpectral):
    r"""Pseudo-spectral solver 3D incompressible Navier-Stokes equations.

    Notes
    -----

    .. |p| mathmacro:: \partial

    .. |vv| mathmacro:: \textbf{v}

    .. |kk| mathmacro:: \textbf{k}

    .. |ek| mathmacro:: \hat{\textbf{e}}_\textbf{k}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    .. |bomega| mathmacro:: \boldsymbol{\omega}

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
      \p_t \hat{\vv} = \hat N(\vv) + L \hat{\vv},

    where

    .. math::

      \hat N(\vv) = -P_\perp \widehat{\vv \cdot \bnabla \vv},

    .. math::

      L = - \nu_\alpha |\kk|^{2\alpha},

    with :math:`P_\perp = (1 - \ek \ek \cdot)` the operator projection on the
    plane perpendicular to the wave number :math:`\kk`. Since the flow is
    incompressible (:math:`\kk \cdot \vv = 0`), the effect of the pressure term
    is taken into account with the operator :math:`P_\perp`.

    In practice, it is more efficient to use the relation

    .. math::

      \vv \cdot \bnabla \vv = \bnabla |\vv|^2/2 - \vv \times \bomega,

    with :math:`\bomega = \bnabla \times \vv` the vorticity, and to compute the
    nonlinear term as

    .. math::

      \hat N(\vv) = P_\perp \widehat{\vv \times \bomega}.

    """
    InfoSolver = InfoSolverNS3D

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        SimulBasePseudoSpectral._complete_params_with_default(params)
        params._set_attribs({"f": None, "no_vz_kz0": False, "projection": None})
        params._set_doc(
            params._doc
            + """
f: float (default None)

    Coriolis parameter (effect of the system rotation).

no_vz_kz0: bool (default False)

    If True, vz(kz=0) is 0.

projection: str (default None)

    If "toroidal" or "vortical", the solution and the equations are projected
    on the toroidal manifold. If "poloidal", on the poloidal one.

"""
        )

    def _init_projection(self):

        try:
            self.no_vz_kz0 = self.params.no_vz_kz0
        except AttributeError:
            self.no_vz_kz0 = False

        if self.no_vz_kz0:
            self.where_kz_0 = np.array(
                abs(self.oper.Kz) == 0.0,
                dtype=np.uint8,
            )

        try:
            projection = self.params.projection
        except AttributeError:
            projection = None

        if projection is None:
            self._projector = self.oper.project_perpk3d
        elif projection in ("toroidal", "vortical"):
            self._projector = self.oper.project_toroidal
        elif projection == "poloidal":
            self._projector = self.oper.project_poloidal
        else:
            raise ValueError(
                "No known projection for params.projection = "
                f"{self.params.projection}"
            )

    def _modif_omegafft_with_f(self, omegax_fft, omegay_fft, omegaz_fft):
        if rank == 0:
            omegaz_fft[0, 0, 0] += self.params.f

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

        omegax_fft = self.state.fields_spect_tmp[0]
        omegay_fft = self.state.fields_spect_tmp[1]
        omegaz_fft = self.state.fields_spect_tmp[2]

        oper.rotfft_from_vecfft_outin(
            vx_fft, vy_fft, vz_fft, omegax_fft, omegay_fft, omegaz_fft
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

        if self.is_forcing_enabled:
            tendencies_fft += self.forcing.get_forcing()

        self.project_state_spect(tendencies_fft)
        self.oper.dealiasing(tendencies_fft)
        return tendencies_fft

    def project_state_spect(self, state_spect):
        vx_fft = state_spect.get_var("vx_fft")
        vy_fft = state_spect.get_var("vy_fft")
        vz_fft = state_spect.get_var("vz_fft")
        self._projector(vx_fft, vy_fft, vz_fft)
        if self.no_vz_kz0:
            dealiasing_variable(vz_fft, self.where_kz_0)
            if "b_fft" in state_spect.keys:
                dealiasing_variable(state_spect.get_var("b_fft"), self.where_kz_0)


if "sphinx" in sys.modules:
    params = Simul.create_default_params()

    __doc__ += (
        "Default parameters\n"
        "------------------\n"
        ".. code-block:: xml\n\n    "
        + "\n    ".join(params.__str__().split("\n\n", 1)[1].split("\n"))
        + "\n"
        + params._get_formatted_docs()
    )


if __name__ == "__main__":

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    n = 32
    L = 4
    params.oper.nx = n
    params.oper.ny = n
    params.oper.nz = n
    params.oper.Lx = L
    params.oper.Ly = L
    params.oper.Lz = L
    params.oper.type_fft = "fluidfft.fft3d.mpi_with_fftwmpi3d"
    # params.oper.type_fft = 'fluidfft.fft3d.with_pyfftw'
    # params.oper.type_fft = 'fluidfft.fft3d.with_cufft'

    delta_x = params.oper.Lx / params.oper.nx
    # params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.nu_8 = 2.0 * 10e-1 * delta_x**8

    params.time_stepping.USE_T_END = True
    params.time_stepping.t_end = 6.0
    # params.time_stepping.it_end = 2

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 1.0
    params.init_fields.noise.length = 1.0

    params.forcing.enable = False
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

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
