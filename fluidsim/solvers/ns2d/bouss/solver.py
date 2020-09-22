"""NS2D Boussinesq solver (:mod:`fluidsim.solvers.ns2d.bouss.solver`)
=====================================================================

.. autoclass:: Simul
   :members:
   :private-members:

"""
import numpy as np

from transonic import jit, Array

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.solvers.ns2d.solver import InfoSolverNS2D, Simul as SimulNS2D


AF = Array[np.float64, "2d"]


@jit
def tendencies_nonlin_ns2dbouss(
    ux: AF, uy: AF, px_rot: AF, py_rot: AF, px_b: AF, py_b: AF
):
    Frot = -ux * px_rot - uy * py_rot + px_b
    Fb = -ux * px_b - uy * py_b
    return Frot, Fb


class InfoSolverNS2DBouss(InfoSolverNS2D):
    def _init_root(self):

        super()._init_root()

        package = "fluidsim.solvers.ns2d.bouss"
        self.module_name = package + ".solver"
        self.class_name = "Simul"
        self.short_name = "ns2d.bouss"

        classes = self.classes

        classes.State.module_name = package + ".state"
        classes.State.class_name = "StateNS2DBouss"


# classes.InitFields.module_name = package + '.init_fields'
# classes.InitFields.class_name = 'InitFieldsNS2DStrat'

# classes.Output.module_name = package + '.output'
# classes.Output.class_name = 'OutputStrat'

# classes.Forcing.module_name = 'fluidsim.solvers.ns2d' + '.forcing'
# classes.Forcing.class_name = 'ForcingNS2D'

# classes.TimeStepping.module_name = \
#     package + '.time_stepping'

# classes.TimeStepping.class_name = 'TimeSteppingPseudoSpectralStrat'


class Simul(SimulNS2D):
    """Pseudo-spectral solver 2D incompressible Navier-Stokes equations."""

    InfoSolver = InfoSolverNS2DBouss

    def tendencies_nonlin(self, state_spect=None, old=None):
        r"""Compute the nonlinear tendencies of the solver ns2d.strat.

        Parameters
        ----------

        state_spect : :class:`fluidsim.base.setofvariables.SetOfVariables`
            optional

            Array containing the state, i.e. the vorticity, in Fourier
            space.  If `state_spect`, the variables vorticity and the
            velocity are computed from it, otherwise, they are taken
            from the global state of the simulation, `self.state`.

            These two possibilities are used during the Runge-Kutta
            time-stepping.

        Returns
        -------

        tendencies_fft : :class:`fluidsim.base.setofvariables.SetOfVariables`
            An array containing the tendencies for the vorticity and the
            buoyancy.

        Notes
        -----

        .. |p| mathmacro:: \partial

        The 2D Navier-Stokes equation can be written

        .. math:: \p_t \hat\zeta = \hat N(\zeta) - \sigma(k) \hat \zeta,

        and

        .. math:: \p_t \hat b = \hat N(b) - \sigma(k) \hat b

        This function compute the nonlinear terms ("tendencies") :math:`\hat
        N(\zeta) = - \mathbf{u}\cdot \mathbf{\nabla} \zeta +
        \mathbf{\nabla}\wedge b\mathbf{e_z} = - \mathbf{u}\cdot \mathbf{\nabla}
        \zeta + \p_x b` and :math:`\hat N(b) = - \mathbf{u}\cdot
        \mathbf{\nabla} b`."""
        oper = self.oper
        fft_as_arg = oper.fft_as_arg
        ifft_as_arg = oper.ifft_as_arg

        if old is None:
            tendencies_fft = SetOfVariables(like=self.state.state_spect)
        else:
            tendencies_fft = old
        Frot_fft = tendencies_fft.get_var("rot_fft")
        Fb_fft = tendencies_fft.get_var("b_fft")

        if state_spect is None:
            rot_fft = self.state.state_spect.get_var("rot_fft")
            b_fft = self.state.state_spect.get_var("b_fft")
            ux = self.state.state_phys.get_var("ux")
            uy = self.state.state_phys.get_var("uy")
        else:
            rot_fft = state_spect.get_var("rot_fft")
            b_fft = state_spect.get_var("b_fft")
            ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
            ux = self.state.field_tmp0
            uy = self.state.field_tmp1
            ifft_as_arg(ux_fft, ux)
            ifft_as_arg(uy_fft, uy)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_b_fft, py_b_fft = oper.gradfft_from_fft(b_fft)

        px_rot = self.state.field_tmp2
        py_rot = self.state.field_tmp3

        px_b = self.state.field_tmp4
        py_b = self.state.field_tmp5

        ifft_as_arg(px_rot_fft, px_rot)
        ifft_as_arg(py_rot_fft, py_rot)
        ifft_as_arg(px_b_fft, px_b)
        ifft_as_arg(py_b_fft, py_b)

        Frot, Fb = tendencies_nonlin_ns2dbouss(ux, uy, px_rot, py_rot, px_b, py_b)

        fft_as_arg(Fb, Fb_fft)
        fft_as_arg(Frot, Frot_fft)

        oper.dealiasing(tendencies_fft)

        # # CHECK NON-LINEAR TRANSFER rot
        # T_rot = np.real(Frot_fft.conj()*rot_fft + Frot_fft*rot_fft.conj())/2.
        # print(('sum(T_rot) = {0:9.4e} ; sum(abs(T_rot)) = {1:9.4e}').format(
        #     self.oper.sum_wavenumbers(T_rot),
        #     self.oper.sum_wavenumbers(abs(T_rot))))

        # # CHECK NON-LINEAR TRANSFER b
        # T_b = np.real(Fb_fft.conj()*b_fft + Fb_fft*b_fft.conj())/2.
        # print(('sum(T_b) = {0:9.4e} ; sum(abs(T_b)) = {1:9.4e}').format(
        #     self.oper.sum_wavenumbers(T_b),
        #     self.oper.sum_wavenumbers(abs(T_b))))

        if self.params.forcing.enable:
            tendencies_fft += self.forcing.get_forcing()

        # CHECK ENERGY CONSERVATION
        # Nrot_fft = tendencies_fft.get_var('rot_fft')
        # Nb_fft = tendencies_fft.get_var('b_fft')
        # self.check_energy_conservation(rot_fft, b_fft, Nrot_fft, Nb_fft)

        return tendencies_fft


# def check_energy_conservation(self, rot_fft, b_fft, N_rot, N_b):
#     """ Check energy conservation for the inviscid case. """
#     oper = self.oper
#     params = self.params

#     # Compute time derivative kinetic energy
#     division = 1./(oper.KX**2 + oper.KY**2)
#     division[np.where(division == np.inf)] = 0

#     pt_energyK_fft = 0.5 * division * np.real(rot_fft.conj() * N_rot)
#     pt_energyK_fft[np.isinf(pt_energyK_fft)] = 0.

#     # Compute time derivative potential energy
#     pt_energyA_fft = (1./(2 * params.N**2)) * np.real(b_fft.conj() * N_b)

#     # Time derivative total energy
#     pt_energy_fft = pt_energyK_fft + pt_energyA_fft

#     # Check time derivative energy is ~ 0.
#     epsilon = 1e-8
#     pt_energy = self.output.sum_wavenumbers(pt_energy_fft)
#     energy_conserved = np.abs(pt_energy) < epsilon
#     if not energy_conserved:
#         print('Energy conserved = ', energy_conserved)
#         print('pt_energy = {}'.format(
#             np.abs(self.output.sum_wavenumbers(pt_energy_fft))))

# def compute_dispersion_relation(self):
#     """
#     Computes the dispersion relation of internal gravity waves solver
#     ns2d.strat.

#     Returns
#     -------
#     omega_dispersion_relation : arr
#       Frequency dispersion relation in rad.
#     """
#     return self.params.N * (self.oper.KX / self.oper.K_not0)


if __name__ == "__main__":

    from math import pi

    import fluiddyn as fld

    params = Simul.create_default_params()

    params.short_name_type_run = "test"

    params.oper.nx = nx = 64
    params.oper.ny = nx // 2
    params.oper.Lx = Lx = 2 * pi
    params.oper.Ly = Lx / 2
    params.oper.coef_dealiasing = 0.5

    delta_x = Lx / nx

    params.nu_2 = 1.0 * 10e-6
    # params.nu_8 = 2.*10e-1*params.forcing.forcing_rate**(1./3)*delta_x**8
    params.time_stepping.USE_CFL = True
    params.time_stepping.USE_T_END = True
    # params.time_stepping.deltat0 = 0.1
    # Period of time of the simulation
    params.time_stepping.t_end = 5.0
    # params.time_stepping.it_end = 50

    params.init_fields.type = "noise"

    params.forcing.enable = True
    # params.forcing.type = 'tcrandom_anisotropic'
    params.forcing.type = "user_defined"

    # params.forcing.nkmax_forcing = 12
    # params.forcing.nkmin_forcing = 4
    # params.forcing.tcrandom_anisotropic.angle = '45'

    # 'Proportional'
    # params.forcing.type_normalize

    params.output.sub_directory = "tests"

    params.output.periods_print.print_stdout = 0.01

    params.output.periods_save.phys_fields = 2.0
    params.output.periods_save.spectra = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.periods_save.spect_energy_budg = 1.0
    params.output.periods_save.increments = 1.0

    params.output.periods_plot.phys_fields = 5.0

    params.output.ONLINE_PLOT_OK = True

    params.output.spectra.HAS_TO_PLOT_SAVED = True
    params.output.spatial_means.HAS_TO_PLOT_SAVED = True
    params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = False
    params.output.increments.HAS_TO_PLOT_SAVED = False

    params.output.phys_fields.field_to_plot = "rot"

    sim = Simul(params)

    # monkey-patching for forcing
    import numpy as np

    forcing_maker = sim.forcing.forcing_maker
    oper = forcing_maker.oper_coarse
    forcing0 = np.cos(2 * np.pi * oper.Y / oper.ly)
    omega = 2 * np.pi

    def compute_forcingc_each_time(self):
        return forcing0 * np.sin(omega * sim.time_stepping.t)

    forcing_maker.monkeypatch_compute_forcingc_each_time(
        compute_forcingc_each_time
    )

    # sim.output.phys_fields.plot()
    sim.time_stepping.start()
    sim.output.phys_fields.plot()

    fld.show()
