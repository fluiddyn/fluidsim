"""Smagorinsky turbulent models (:mod:`fluidsim.base.turb_model.smagorinsky`)
=============================================================================

Provides:

.. autoclass:: SmagorinskyModel
   :members:
   :private-members:

"""

from math import sqrt

from fluidsim.base.turb_model.base import SpecificTurbModelSpectral
from fluidsim.base.turb_model.stress_tensor import StressTensorComputer3D


class SmagorinskyModel(SpecificTurbModelSpectral):
    r"""Smagorinsky turbulence model

    .. |p| mathmacro:: \partial

    .. |Sij| mathmacro:: \bar{S}_{ij}

    .. math::

      \p_t v_i = ... + \p_j(2 \nu_T \Sij),

    where :math:`\Sij = (\p_i v_j + \p_j v_i) / 2`. The turbulent viscosity
    :math:`\nu_T` is computed for this model as

    .. math::

      \nu_T = C \Delta^2 \sqrt{2 \Sij \Sij},

    with :math:`C = 0.18` and :math:`\Delta = L_x / n_x`.

    """
    _module_name = "fluidsim.base.turb_model.smagorinsky"
    tag = "smagorinsky"

    @classmethod
    def complete_params_with_default(cls, params):
        params.turb_model._set_child(cls.tag, attribs={"C": 0.18})

    def __init__(self, sim):
        super().__init__(sim)
        self.stress_tensor = StressTensorComputer3D(sim.oper)

        C = sim.params.turb_model.smagorinsky.C
        delta = sim.params.oper.Lx / sim.params.oper.nx

        self.C_nu_T = C * delta**2 * sqrt(2)

    def get_forcing(self, **kwargs):

        ux_fft = kwargs["vx_fft"]
        uy_fft = kwargs["vy_fft"]
        uz_fft = kwargs["vz_fft"]

        Sxx, Syy, Szz, Syx, Szx, Szy = self.stress_tensor.compute_stress_tensor(
            ux_fft, uy_fft, uz_fft
        )
        norm = self.stress_tensor.compute_norm(Sxx, Syy, Szz, Syx, Szx, Szy)

        nu_T_2 = 2 * self.C_nu_T * norm

        nuT_2_Sxx = nu_T_2 * Sxx
        nuT_2_Syy = nu_T_2 * Syy
        nuT_2_Szz = nu_T_2 * Szz

        nuT_2_Syx = nu_T_2 * Syx
        nuT_2_Szx = nu_T_2 * Szx
        nuT_2_Szy = nu_T_2 * Szy

        oper = self.sim.oper
        fft = oper.fft

        nuT_2_Sxx_fft = fft(nuT_2_Sxx)
        nuT_2_Syy_fft = fft(nuT_2_Syy)
        nuT_2_Szz_fft = fft(nuT_2_Szz)

        nuT_2_Syx_fft = fft(nuT_2_Syx)
        nuT_2_Szx_fft = fft(nuT_2_Szx)
        nuT_2_Szy_fft = fft(nuT_2_Szy)

        # using symmetry of Sij
        nuT_2_Sxy_fft = nuT_2_Syx_fft
        nuT_2_Sxz_fft = nuT_2_Szx_fft
        nuT_2_Syz_fft = nuT_2_Szy_fft

        Kx = oper.Kx
        Ky = oper.Ky
        Kz = oper.Kz

        fx_fft = 2j * (
            Kx * nuT_2_Sxx_fft + Ky * nuT_2_Sxy_fft + Kz * nuT_2_Sxz_fft
        )
        fy_fft = 2j * (
            Kx * nuT_2_Syx_fft + Ky * nuT_2_Syy_fft + Kz * nuT_2_Syz_fft
        )
        fz_fft = 2j * (
            Kx * nuT_2_Szx_fft + Ky * nuT_2_Szy_fft + Kz * nuT_2_Szz_fft
        )

        self.forcing_fft.set_var("vx_fft", fx_fft)
        self.forcing_fft.set_var("vy_fft", fy_fft)
        self.forcing_fft.set_var("vz_fft", fz_fft)

        return self.forcing_fft
