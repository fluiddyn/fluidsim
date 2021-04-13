# -*- coding: utf-8 -*-
"""Output (:mod:`fluidsim.solvers.ns2d.strat.output`)
=====================================================

Provides the modules:

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spectra
   spect_energy_budget

and the main output class for the ns2d.strat solver:

.. autoclass:: OutputStrat
   :members:
   :private-members:

"""

import numpy as np

from math import radians

from fluidsim.solvers.ns2d.output import Output


class OutputStrat(Output):
    """Output for ns2d.strat solver."""

    def __init__(self, sim):
        self.sim = sim
        if sim.params.forcing.type.endswith("anisotropic"):
            self._init_froude_number()
            self.ratio_omegas = self._compute_ratio_omegas()

        super().__init__(sim)

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        Output._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = "fluidsim.solvers.ns2d.strat.output"

        classes.PrintStdOut.module_name = base_name_mod + ".print_stdout"
        classes.PrintStdOut.class_name = "PrintStdOutNS2DStrat"

        classes.PhysFields.module_name = base_name_mod + ".phys_fields"
        classes.PhysFields.class_name = "PhysFields2DStrat"

        classes.Spectra._set_attribs(
            {
                "module_name": base_name_mod + ".spectra",
                "class_name": "SpectraNS2DStrat",
            }
        )

        classes.SpectraMultiDim._set_attribs(
            {
                "module_name": base_name_mod + ".spectra_multidim",
                "class_name": "SpectraMultiDimNS2DStrat",
            }
        )

        classes.SpatialMeans._set_attribs(
            {
                "module_name": base_name_mod + ".spatial_means",
                "class_name": "SpatialMeansNS2DStrat",
            }
        )

        classes.SpectEnergyBudg._set_attribs(
            {
                "module_name": base_name_mod + ".spect_energy_budget",
                "class_name": "SpectralEnergyBudgetNS2DStrat",
            }
        )

    def compute_energies_fft(self):
        """Compute the kinetic and potential energy (k)"""
        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        b_fft = self.sim.state.state_spect.get_var("b_fft")
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyK_fft = (np.abs(ux_fft) ** 2 + np.abs(uy_fft) ** 2) / 2

        if self.sim.params.N == 0:
            energyA_fft = np.zeros_like(energyK_fft)
        else:
            energyA_fft = ((np.abs(b_fft) / self.sim.params.N) ** 2) / 2
        return energyK_fft, energyA_fft

    def compute_energies2_fft(self):
        """Compute the two kinetic energies for u_x and u_y"""
        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        energyK_ux_fft = (np.abs(ux_fft) ** 2) / 2
        energyK_uy_fft = (np.abs(uy_fft) ** 2) / 2
        return energyK_ux_fft, energyK_uy_fft

    def compute_energy_fft(self):
        """Compute energy(k)"""
        energyK_fft, energyA_fft = self.compute_energies_fft()
        return energyK_fft + energyA_fft

    def compute_enstrophy_fft(self):
        """Compute enstrophy(k)"""
        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        return np.abs(rot_fft) ** 2 / 2

    def compute_energy(self):
        """Compute the spatially averaged energy."""
        energy_fft = self.compute_energy_fft()
        return self.sum_wavenumbers(energy_fft)

    def compute_energies(self):
        """Compute the kinetic and potential energy"""
        energyK_fft, energyA_fft = self.compute_energies_fft()
        energyK_ux_fft, energyK_uy_fft = self.compute_energies2_fft()
        energyK = self.sum_wavenumbers(energyK_fft)
        energyA = self.sum_wavenumbers(energyA_fft)

        # energyK_ux (##TODO: remove it. not sense)
        energyK_ux = self.sum_wavenumbers(energyK_ux_fft)
        return energyK, energyA, energyK_ux

    def compute_enstrophy(self):
        """Compute the spatially averaged enstrophy."""
        enstrophy_fft = self.compute_enstrophy_fft()
        return self.sum_wavenumbers(enstrophy_fft)

    def _init_froude_number(self):
        """Compute froude number ONLY for anisotropic forcing."""
        angle = self.sim.params.forcing.tcrandom_anisotropic.angle
        if isinstance(angle, str):
            if angle.endswith("°"):
                angle = radians(float(angle[:-1]))
            else:
                raise ValueError(
                    "Angle should be a string with \n"
                    + "the degree symbol or a float in radians"
                )

        self.froude_number = np.sin(angle)

    def _compute_ratio_omegas(self):
        r"""Compute ratio omegas; gamma = \omega_l / \omega_{af}"""
        params = self.sim.params
        pforcing = params.forcing
        forcing_rate = pforcing.forcing_rate

        # Compute forcing wave-number
        nkmax_forcing = pforcing.nkmax_forcing
        nkmin_forcing = pforcing.nkmin_forcing

        deltak = max(2 * np.pi / params.oper.Lx, 2 * np.pi / params.oper.Ly)
        k_f = ((nkmax_forcing + nkmin_forcing) / 2) * deltak
        l_f = 2 * np.pi / k_f

        # Compute linear frequency in s^-1
        omega_l = params.N * self.froude_number
        omega_l = omega_l / (2 * np.pi)

        # Compute forcing frequency
        if pforcing.normalized.constant_rate_of is None:
            if pforcing.key_forced is None or pforcing.key_forced == "rot_fft":
                omega_af = forcing_rate ** (1.0 / 3)

            elif pforcing.key_forced == "ap_fft":
                omega_af = forcing_rate ** (1.0 / 7) * l_f ** (-2.0 / 7)

            else:
                raise ValueError("params.forcing.key_forced is not known.")

        elif pforcing.normalized.constant_rate_of in ["energy", "energyK"]:
            omega_af = forcing_rate ** (1.0 / 3) * l_f ** (-2.0 / 3)

        else:
            raise ValueError(f"{pforcing.normalized.constant_rate_of} not known.")

        return omega_l / omega_af

    def _init_sim_repr_maker(self):
        """initialize sim_repr_maker."""
        sim_repr_maker = super()._init_sim_repr_maker()
        if self.sim.params.forcing.type.endswith("anisotropic"):
            sim_repr_maker.add_parameters(
                {"F": self.froude_number, "gamma": self.ratio_omegas},
                formats={"gamma": ".1f"},
            )
        return sim_repr_maker

    def plot_summary(self, field="b", time_phys=None, tmin=None, tmax=None):
        """
        Plots summary of all outputs of a simulation.
        """
        # pylint: disable=maybe-no-member
        self.phys_fields.plot(field=field, time=time_phys, QUIVER=False)
        self.print_stdout.plot()
        self.spatial_means.plot()
        self.spectra.plot1d(tmin=tmin, tmax=tmax, level3=10.0)
        self.spect_energy_budg.plot(tmin=tmin, tmax=tmax)
