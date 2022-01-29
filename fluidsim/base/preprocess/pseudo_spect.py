"""Preprocessing for pseudo-spectral solvers (:mod:`fluidsim.base.preprocess.pseudo_spect`)
===========================================================================================

Provides:

.. autoclass:: PreprocessPseudoSpectral
   :members:
   :private-members:


"""

import numpy as np
from .base import PreprocessBase


class PreprocessPseudoSpectral(PreprocessBase):
    _tag = "pseudo_spectral"

    def __call__(self):
        """Preprocesses if enabled."""

        super().__call__()
        if self.params.enable:
            if self.sim.params.forcing.enable:
                if "forcing" in self.params.init_field_scale:
                    self.set_forcing_rate()
                    self.normalize_init_fields()
                else:
                    self.normalize_init_fields()
                    self.set_forcing_rate()
            else:
                self.normalize_init_fields()

            self.sim.state.clear_computed()
            self.set_viscosity()
            self.output._save_info_solver_params_xml(replace=True)

    def normalize_init_fields(self):
        """
        A non-dimensionalization procedure for the initialized fields.

        Parameters
        ----------------
        init_field_scale : string (use 'energy', 'unity')
            Set quantity to normalize initialized fields with.
        """
        state = self.sim.state
        scale = self.params.init_field_scale
        C = float(self.params.init_field_const)

        if scale == "energy":
            try:
                (Ek,) = self.output.compute_quad_energies()
            except:
                Ek = self.output.compute_energy()

            ux_fft = state.get_var("ux_fft")
            uy_fft = state.get_var("uy_fft")

            if Ek != 0.0:
                ux_fft = (C / Ek) ** 0.5 * ux_fft
                uy_fft = (C / Ek) ** 0.5 * uy_fft

            try:
                state.init_from_uxuyfft(ux_fft, uy_fft)
            except AttributeError:
                rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
                state.init_statespect_from(rot_fft=rot_fft)
        elif scale == "enstrophy":
            omega_0 = self.output.compute_enstrophy()
            rot_fft = state.get_var("rot_fft")

            if omega_0 != 0.0:
                rot_fft = (C / omega_0) ** 0.5 * rot_fft
                state.init_from_rotfft(rot_fft)

        elif scale == "enstrophy_forcing":
            P = self.sim.params.forcing.forcing_rate
            k_f = self.oper.deltak * (
                (
                    self.sim.params.forcing.nkmax_forcing
                    + self.sim.params.forcing.nkmin_forcing
                )
                // 2
            )
            omega_0 = self.output.compute_enstrophy()
            rot_fft = state.get_var("rot_fft")

            if omega_0 != 0.0:
                C_0 = omega_0 / (P ** (2.0 / 3) * k_f ** (4.0 / 3))
                rot_fft = (C / C_0) ** 0.5 * rot_fft
                state.init_from_rotfft(rot_fft)

        elif scale == "unity":
            pass
        else:
            raise ValueError("Unknown initial fields scaling: ", scale)

    def set_viscosity(self):
        r"""Based on

        - the initial total enstrophy, \Omega_0, or

        - the initial energy

        - the forcing rate, \epsilon

        the viscosity scale or Reynolds number is set.

        Parameters
        ----------

        params.preprocess.viscosity_type : string
          Type/Order of viscosity desired

        params.preprocess.viscosity_scale : string
          Mean quantity to be scaled against

        params.preprocess.viscosity_const : float
          Calibration constant to set dissipative wave number

        Notes
        -----

        Algorithm: Sets viscosity variable nu and reinitializes f_d array for
        timestepping

        """
        params = self.params
        viscosity_type = params.viscosity_type
        viscosity_scale = params.viscosity_scale
        C = params.viscosity_const

        if viscosity_scale == "enstrophy":
            args = [self.output.compute_enstrophy()]
        elif viscosity_scale == "energy":
            args = [self.output.compute_energy()]
        elif viscosity_scale == "enstrophy_forcing":
            args = [
                self.output.compute_enstrophy(),
                self.sim.params.forcing.forcing_rate,
            ]
        elif viscosity_scale == "forcing":
            args = [self.sim.params.forcing.forcing_rate]
        else:
            raise ValueError("Unknown viscosity scale: %s" % viscosity_scale)

        result = calcul_viscosity(
            C,
            viscosity_scale,
            viscosity_type,
            oper=self.oper,
            verbose=False,
            *args,
        )
        for v in result.values():
            attr, order, nu = v
            self.sim.params.__setattr__(attr, nu)

        self.sim.time_stepping.__init__(self.sim)

    def set_forcing_rate(self):
        r"""Based on C, a non-dimensional ratio of forcing rate to one of the
        following forcing scales

        - the initial total energy, math:: E_0
        - the initial total enstrophy, math:: \Omega_0

        the forcing rate is set.

        Parameters
        ----------

        params.preprocess.forcing_const : float
          Non-dimensional ratio of forcing_scale to forcing_rate

        params.preprocess.forcing_scale : string
          Mean quantity to be scaled against

        """
        params = self.params

        forcing_scale = params.forcing_scale
        C = float(params.forcing_const)
        # Forcing wavenumber
        k_f = self.oper.deltak * (
            (
                self.sim.params.forcing.nkmax_forcing
                + self.sim.params.forcing.nkmin_forcing
            )
            // 2
        )

        if forcing_scale == "unity":
            self.sim.params.forcing.forcing_rate = C
        elif forcing_scale == "energy":
            energy_0 = self.output.compute_energy()
            self.sim.params.forcing.forcing_rate = C * energy_0**1.5 * k_f
        elif forcing_scale == "enstrophy":
            omega_0 = self.output.compute_enstrophy()
            self.sim.params.forcing.forcing_rate = C * omega_0**1.5 / k_f**2
        else:
            raise ValueError("Unknown forcing scale: %s" % forcing_scale)

        self.sim.forcing.__init__(self.sim)


def calcul_viscosity(
    viscosity_const, viscosity_scale, viscosity_type, *args, **kwargs
):
    """Calculates viscosity based on scaling formulae.  Use this function
    to estimate viscosity before runtime.

    Parameters
    ----------
    viscosity_const : scalar
      Calibration constant to set dissipative wave number

    viscosity_scale : string
      Mean quantity to be scaled against

    viscosity_type : string
      Type/Order of viscosity desired

    *args : scalar(s)
      Estimated value for viscosity scale

    oper : object of :class:`fluidsim.operators.operators.Operators`, optional

    nh, Lh, coef_dealiasing, nk_f : scalars, optional
        No. of grid points, length of the box, coeff. of dealiasing and
        forcing wavenumber index.

    verbose : bool, optional
        For verbose output


    Examples
    --------
    >>> calcul_viscosity(
    ...     1, 'enstrophy', 'laplacian', 50., oper=sim.oper)
    >>> calcul_viscosity(
    ...     0.5, 'forcing', 'laplacian_hyper8', oper=sim.oper, verbose=False)
    >>> calcul_viscosity(
    ...     1, 'enstrophy_forcing', 'laplacian', 50., 1.,
    ...     nh=128, Lh=50, coef_dealiasing=2. / 3, nk_f=6)
    >>> calcul_viscosity(
    ...     0.785, 'forcing', 'hyper8', 1.,
    ...     nh=1920, Lh=50, coef_dealiasing=8. / 9, nk_f=6)


    """
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = True

    if "oper" in kwargs:
        oper = kwargs["oper"]
        coef_dealiasing = oper.coef_dealiasing
        nk_f = (
            oper.params.forcing.nkmax_forcing + oper.params.forcing.nkmin_forcing
        ) // 2
        delta_x = oper.deltax
        deltak = oper.deltak
    else:
        nh = kwargs["nh"]
        Lh = kwargs["Lh"]
        coef_dealiasing = kwargs["coef_dealiasing"]
        nk_f = kwargs["nk_f"]
        delta_x = Lh / nh
        deltak = 2 * np.pi / Lh

    # Smallest resolved scale
    k_max = np.pi / delta_x * coef_dealiasing
    # OR np.pi / k_d, the dissipative wave number
    C = viscosity_const
    length_scale = C * np.pi / k_max
    k_f = deltak * nk_f
    large_scale = np.pi / k_f

    k_diss = k_max / C / np.pi
    if verbose:
        print("Max. wavenumber =", np.pi / delta_x)
        print("Max. resolved wavenumber, k_max =", k_max)
        print("Grid spacing, delta_x =", delta_x)
        print("\nESTIMATED (P~eps)")
        print(
            f"Dissipation wavenumber, k_d = {k_diss}; k_d / k_f = {k_diss / k_f}"
        )
        print(
            f"Dissipation length scale, L_d = {length_scale}; L_d / L_f = {length_scale / large_scale}"
        )
        print("Viscosity scale:", viscosity_scale, "=", args)

    if len(args) == 0:
        raise ValueError("Expected values related to `viscosity_scale` as *args")

    if viscosity_scale == "enstrophy":
        omega_0 = args[0]
        eta = omega_0**1.5
        time_scale = eta ** (-1.0 / 3)
    elif viscosity_scale == "energy":
        energy_0 = args[0]
        epsilon = energy_0 * (1.5) / large_scale
        time_scale = epsilon ** (-1.0 / 3) * length_scale ** (2.0 / 3)
    elif viscosity_scale == "enstrophy_forcing":
        omega_0 = args[0]
        eta = omega_0**1.5
        t1 = eta ** (-1.0 / 3)
        # Energy dissipation rate
        epsilon = args[1]
        t2 = epsilon ** (-1.0 / 3) * length_scale ** (2.0 / 3)
        time_scale = min(t1, t2)
    elif viscosity_scale == "forcing":
        epsilon = args[0]
        time_scale = epsilon ** (-1.0 / 3) * length_scale ** (2.0 / 3)
    else:
        raise ValueError("Unknown viscosity scale: %s" % viscosity_scale)

    dict_visc = {
        "laplacian": ["nu_2", 2],
        "hyper4": ["nu_4", 4],
        "hyper8": ["nu_8", 8],
        "hypo": ["nu_m4", -4],
    }

    if verbose:
        if "oper" in kwargs:
            epsilon = oper.params.forcing.forcing_rate
        elif "eps" in kwargs:
            epsilon = kwargs["eps"]
        else:
            epsilon = 1.0

        print("Dissipation, epsilon =", epsilon)
        kolmo_len = []

    if not any([k in viscosity_type for k in dict_visc]):
        raise ValueError("Unknown viscosity type: %s" % viscosity_type)

    else:
        for k, v in dict_visc.items():
            if k in viscosity_type:
                attr, order = v
                nu = length_scale**order / time_scale
                v.append(nu)
                if verbose:
                    kolmo_len.append(
                        (nu**3 / epsilon) ** (1.0 / (3 * order - 2))
                    )
            else:
                v.append(0.0)

    if verbose:
        length_scale = np.mean(kolmo_len)
        k_diss = 1.0 / length_scale
        print(f"\nCALCULATED (eps={epsilon})")
        print(
            f"Dissipation wavenumber, k_d = {k_diss}; k_d / k_f = {k_diss / k_f}"
        )
        print(
            f"Dissipation length scale, L_d = {length_scale}; L_d / L_f = {length_scale / large_scale}"
        )

    return dict_visc
