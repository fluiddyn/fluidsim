"""Output SW1L (:mod:`fluidsim.solvers.sw1l.output`)
====================================================

.. autoclass:: OutputBaseSW1L
   :members:
   :private-members:

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spect_energy_budget
   spectra
   normal_mode

"""

import numpy as np

from transonic import jit

from fluidsim.base.output import OutputBasePseudoSpectral


@jit
def linear_eigenmode_from_values_1k(
    ux_fft: np.complex128,
    uy_fft: np.complex128,
    eta_fft: np.complex128,
    kx: float,
    ky: float,
    f: "float or int",
    c2: "float or int",
):
    """Compute q, d, a (fft) for a single wavenumber."""
    div_fft = 1j * (kx * ux_fft + ky * uy_fft)
    rot_fft = 1j * (kx * uy_fft - ky * ux_fft)
    q_fft = rot_fft - f * eta_fft
    k2 = kx**2 + ky**2
    ageo_fft = f * rot_fft / c2 + k2 * eta_fft
    return q_fft, div_fft, ageo_fft


class OutputBaseSW1L(OutputBasePseudoSpectral):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer *info_solver* with child classes to be
        instantiated under *sim.output*.

        This is a static method!
        """
        classes = info_solver.classes.Output._set_child("classes")

        package = "fluidsim.solvers.sw1l.output"

        classes._set_child(
            "PrintStdOut",
            attribs={
                "module_name": package + ".print_stdout",
                "class_name": "PrintStdOutSW1L",
            },
        )

        classes._set_child(
            "PhysFields",
            attribs={
                "module_name": "fluidsim.base.output.phys_fields2d",
                "class_name": "PhysFieldsBase2D",
            },
        )

        classes._set_child(
            "Spectra",
            attribs={
                "module_name": package + ".spectra",
                "class_name": "SpectraSW1LNormalMode",
            },
        )

        classes._set_child(
            "SpatialMeans",
            attribs={
                "module_name": package + ".spatial_means",
                "class_name": "SpatialMeansSW1L",
            },
        )

        attribs = {
            "module_name": package + ".spect_energy_budget",
            "class_name": "SpectralEnergyBudgetSW1L",
        }
        classes._set_child("SpectralEnergyBudget", attribs=attribs)

        attribs = {
            "module_name": package + ".increments",
            "class_name": "IncrementsSW1L",
        }
        classes._set_child("Increments", attribs=attribs)

        attribs = {
            "module_name": "fluidsim.base.output.prob_dens_func",
            "class_name": "ProbaDensityFunc",
        }
        classes._set_child("ProbaDensityFunc", attribs=attribs)

        attribs = {
            "module_name": "fluidsim.base.output.time_signals_fft",
            "class_name": "TimeSignalsK",
        }
        classes._set_child("TimeSignalsK", attribs=attribs)

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        OutputBasePseudoSpectral._complete_params_with_default(
            params, info_solver
        )

        params.output.phys_fields.field_to_plot = "rot"

    def linear_eigenmode_from_values_1k(self, ux_fft, uy_fft, eta_fft, kx, ky):
        """Compute the linear eigenmodes for a single wavenumber."""
        return linear_eigenmode_from_values_1k(
            ux_fft, uy_fft, eta_fft, kx, ky, self.sim.params.f, self.sim.params.c2
        )

    def omega_from_wavenumber(self, k):
        r"""Evaluates the dispersion relation and returns the linear frequency

        .. math:: \omega = \sqrt{f ^ 2 + (ck)^2}

        """
        return np.sqrt(self.sim.params.f**2 + self.sim.params.c2 * k**2)

    def compute_enstrophy_fft(self):
        r"""Calculate enstrophy from vorticity in the spectral space."""
        rot_fft = self.sim.state.get_var("rot_fft")
        return np.abs(rot_fft) ** 2 / 2.0

    def compute_PV_fft(self):
        r"""Compute Ertel and Charney (QG) potential vorticity.

        .. math:: \zeta_{er} = \frac{f + \zeta}{1 + \eta}
        .. math:: \zeta_{ch} = \zeta - f \eta

        """
        get_var = self.sim.state.get_var
        rot = get_var("rot")
        eta = self.sim.state.state_phys.get_var("eta")
        ErtelPV_fft = self.oper.fft2((self.sim.params.f + rot) / (1.0 + eta))
        rot_fft = get_var("rot_fft")
        eta_fft = get_var("eta_fft")
        CharneyPV_fft = rot_fft - self.sim.params.f * eta_fft
        return ErtelPV_fft, CharneyPV_fft

    def compute_PE_fft(self):
        """Compute Ertel and Charney (QG) potential enstrophy."""
        ErtelPV_fft, CharneyPV_fft = self.compute_PV_fft()
        return (abs(ErtelPV_fft) ** 2 / 2.0, abs(CharneyPV_fft) ** 2 / 2.0)

    def compute_CharneyPE_fft(self):
        """Compute Charney (QG) potential enstrophy."""
        rot_fft = self.sim.state.get_var("rot_fft")
        eta_fft = self.sim.state.get_var("eta_fft")
        CharneyPV_fft = rot_fft - self.sim.params.f * eta_fft
        return abs(CharneyPV_fft) ** 2 / 2.0

    def compute_energies(self):
        """Compute kinetic, available potential and rotational kinetic energies."""
        energyK_fft, energyA_fft, energyKr_fft = self.compute_energies_fft()
        return (
            self.sum_wavenumbers(energyK_fft),
            self.sum_wavenumbers(energyA_fft),
            self.sum_wavenumbers(energyKr_fft),
        )

    def compute_energiesKA(self):
        """Compute K.E. and A.P.E."""
        energyK_fft, energyA_fft = self.compute_energiesKA_fft()
        return (
            self.sum_wavenumbers(energyK_fft),
            self.sum_wavenumbers(energyA_fft),
        )

    def compute_energy(self):
        """Compute total energy by summing K.E. and A.P.E."""
        energyK_fft, energyA_fft = self.compute_energiesKA_fft()
        return self.sum_wavenumbers(energyK_fft) + self.sum_wavenumbers(
            energyA_fft
        )

    def compute_enstrophy(self):
        """Compute total enstrophy."""
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)

    def compute_lin_energies_fft(self):
        r"""Compute quadratic energies decomposed into contributions from
        potential vorticity (:math:`q`), divergence (:math:`\nabla.\mathbf u`),
        and ageostrophic variable (:math:`a`).

        """
        get_var = self.sim.state.get_var
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        eta_fft = get_var("eta_fft")

        q_fft, div_fft, ageo_fft = self.oper.qdafft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
        energy_dlin_fft = 0.5 * (np.abs(udx_fft) ** 2 + np.abs(udy_fft) ** 2)

        ugx_fft, ugy_fft, etag_fft = self.oper.uxuyetafft_from_qfft(q_fft)
        energy_glin_fft = 0.5 * (
            np.abs(ugx_fft) ** 2
            + np.abs(ugy_fft) ** 2
            + self.sim.params.c2 * np.abs(etag_fft) ** 2
        )

        uax_fft, uay_fft, etaa_fft = self.oper.uxuyetafft_from_afft(ageo_fft)
        energy_alin_fft = 0.5 * (
            np.abs(uax_fft) ** 2
            + np.abs(uay_fft) ** 2
            + self.sim.params.c2 * np.abs(etaa_fft) ** 2
        )

        return energy_glin_fft, energy_dlin_fft, energy_alin_fft


class OutputSW1L(OutputBaseSW1L):
    def compute_energies_fft(self):
        r"""Compute kinetic, available potential and rotational kinetic energies
        in the spectral space.

        .. math:: E_K = (h \mathbf u).\mathbf u / 2
        .. math:: E_A = c^2 \eta^2) / 2
        .. math:: E_{K,r} = (h \mathbf u_r).\mathbf u_r / 2

        """
        get_var = self.sim.state.get_var
        eta_fft = get_var("eta_fft")
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft) ** 2 / 2
        Jx_fft = get_var("Jx_fft")
        Jy_fft = get_var("Jy_fft")
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        energyK_fft = (
            np.real(Jx_fft.conj() * ux_fft + Jy_fft.conj() * uy_fft) / 2.0
        )

        rot_fft = get_var("rot_fft")
        uxr_fft, uyr_fft = self.oper.vecfft_from_rotfft(rot_fft)
        rotJ_fft = self.oper.rotfft_from_vecfft(Jx_fft, Jy_fft)
        Jxr_fft, Jyr_fft = self.oper.vecfft_from_rotfft(rotJ_fft)
        energyKr_fft = (
            np.real(Jxr_fft.conj() * uxr_fft + Jyr_fft.conj() * uyr_fft) / 2.0
        )
        return energyK_fft, energyA_fft, energyKr_fft

    def compute_energiesKA_fft(self):
        """Compute K.E. and A.P.E in the spectral space."""
        get_var = self.sim.state.get_var
        eta_fft = get_var("eta_fft")
        energyA_fft = self.sim.params.c2 * np.abs(eta_fft) ** 2 / 2
        Jx_fft = get_var("Jx_fft")
        Jy_fft = get_var("Jy_fft")
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")
        energyK_fft = (
            np.real(Jx_fft.conj() * ux_fft + Jy_fft.conj() * uy_fft) / 2.0
        )

        return energyK_fft, energyA_fft
