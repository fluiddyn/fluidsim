# -*- coding: utf-8 -*-

"""Stratified NS3D solver (:mod:`fluidsim.solvers.ns3d.strat.solver`)
=====================================================================

.. autoclass:: InfoSolverNS3DStrat
   :members:
   :private-members:

.. autoclass:: Simul
   :members:
   :private-members:

"""

from transonic import boost

from fluidfft.fft3d.operators import vector_product

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.operators.operators3d import dealiasing_variable

from ..solver import InfoSolverNS3D, Simul as SimulNS3D


Ac = "complex128[:,:,:]"


@boost
def compute_fb_fft(div_vb_fft: Ac, N: "float or int", vz_fft: Ac):
    fb_fft = div_vb_fft
    fb_fft[:] = -div_vb_fft - N**2 * vz_fft
    return fb_fft


class InfoSolverNS3DStrat(InfoSolverNS3D):
    def _init_root(self):

        super()._init_root()

        package = "fluidsim.solvers.ns3d.strat"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "ns3d.strat"

        classes = self.classes

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateNS3DStrat"

        # classes.InitFields.module_name = package + '.init_fields'
        # classes.InitFields.class_name = 'InitFieldsNS3D'

        classes.Output.module_name = package + ".output"
        classes.Output.class_name = "Output"


# classes.Forcing.module_name = package + '.forcing'
# classes.Forcing.class_name = 'ForcingNS3D'


class Simul(SimulNS3D):
    r"""Pseudo-spectral solver 3D incompressible Navier-Stokes equations.

    Notes
    -----

    .. |p| mathmacro:: \partial

    .. |vv| mathmacro:: \textbf{v}

    .. |kk| mathmacro:: \textbf{k}

    .. |ek| mathmacro:: \textbf{e}_\textbf{k}

    .. |ez| mathmacro:: \textbf{e}_\textbf{z}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    .. |bomega| mathmacro:: \boldsymbol{\omega}

    This class is dedicated to solve with a pseudo-spectral method the
    incompressible Navier-Stokes equations (possibly with hyper-viscosity)
    under the Boussinesq approximation with a constant Brunt-Vaisala frequency:

    .. math::

      \p_t \vv + \vv \cdot \bnabla \vv =
      - \bnabla p + b \ez - \nu_\alpha (-\Delta)^{\alpha/2} \vv,

      \p_t b   + \vv \cdot \bnabla b =
      - N^2 v_z  - \nu_\alpha (-\Delta)^{\alpha/2} b,

    where :math:`\vv` is the non-divergent velocity (:math:`\bnabla \cdot \vv =
    0`), :math:`p` is the pressure, :math:`\Delta` is the 3D Laplacian
    operator, :math:`b` is the buoyancy and :math:`N` is the (constant)
    Brunt-Vaisala frequency.

    The equation for the velocity can be rewritten as (here without the viscous term)

    .. math::

      \p_t \vv =
      - \bnabla (p + |\vv|^2/2) + \vv \times \bomega  + b \ez,

    In Fourier space, we obtain

    .. math::
      \p_t \hat{\vv} = \hat N(\vv) + L \hat{\vv},

    where

    .. math::

      \hat N(\vv) = P_\perp \widehat{\vv \times \bomega} + \hat b \ez,

    .. math::

      L = - \nu_\alpha |\kk|^{2\alpha},

    with :math:`P_\perp = (1 - \ek \ek \cdot)` the operator projection on the
    plane perpendicular to the wave number :math:`\kk`.

    """
    InfoSolver = InfoSolverNS3DStrat

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        SimulNS3D._complete_params_with_default(params)
        attribs = {"N": 1.0}
        params._set_attribs(attribs)

    @classmethod
    def _modify_sim_repr_maker(cls, sim_repr_maker):
        sim_repr_maker.add_parameters({"N": sim_repr_maker.sim.params.N})

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

        div_vb_fft = oper.div_vb_fft_from_vb(vx, vy, vz, b)
        fb_fft = compute_fb_fft(div_vb_fft, self.params.N, vz_fft)

        tendencies_fft.set_var("b_fft", fb_fft)

        if self.is_forcing_enabled:
            tendencies_fft += self.forcing.get_forcing()

        self.project_state_spect(tendencies_fft)
        self.oper.dealiasing(tendencies_fft)
        return tendencies_fft

    def compute_dispersion_relation(self):
        """
        Computes the pulsation of internal gravity waves solver
        ns3d.strat.

        Returns
        -------
        omega_dispersion_relation : arr
        pulsation in rad.
        """
        return self.params.N * np.sqrt(
            (self.oper.Kx**2 + self.oper.Ky**2)
            * self.oper.inv_K_square_nozero
        )


if __name__ == "__main__":

    import numpy as np

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    n = 16
    L = 2 * np.pi
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
    params.time_stepping.it_end = 2

    params.init_fields.type = "noise"
    params.init_fields.noise.velo_max = 1.0
    params.init_fields.noise.length = 1.0

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

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    # sim.output.phys_fields.plot()

    fld.show()
