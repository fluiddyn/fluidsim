"""Time stepping (:mod:`fluidsim.solvers.ns2d.strat.time_stepping`)
===============================================================================


Provides:

.. autoclass:: TimeSteppingPseudoSpectralStrat
   :members:
   :private-members:

"""

from __future__ import print_function

import numpy as np

from math import pi
from fluiddyn.util import mpi
from fluidsim.base.time_stepping.pseudo_spect_cy import \
    TimeSteppingPseudoSpectral


class TimeSteppingPseudoSpectralStrat(TimeSteppingPseudoSpectral):
    """
    Class time stepping for the solver ns2d.strat.

    """

    def _init_compute_time_step(self):
        """
        Initialization compute time step solver ns2d.strat.
        """
        super(TimeSteppingPseudoSpectralStrat, self)._init_compute_time_step()

        # Coefficients dt
        self.CFL = 1.
        self.coef_deltat_dispersion_relation = 1.0
        self.coef_group = 1.0
        self.coef_phase = 1.0

        has_vars = self.sim.state.has_vars
        has_ux = (has_vars('ux') or has_vars('vx'))
        has_uy = (has_vars('uy') or has_vars('vy'))
        has_b = has_vars('b')

        if has_ux and has_uy and has_b:
            self._compute_time_increment_CLF = \
                self._compute_time_increment_CFL_uxuyb

        # Try to compute deltat_dispersion_relation.
        try:
            self.dispersion_relation = self.sim.compute_dispersion_relation()
        except AttributeError:
            print('compute_dispersion_relation is not'
                  'not implemented.')
            self.deltat_dispersion_relation = 1.
        else:
            freq_disp_relation = self.dispersion_relation.max()

            if mpi.nb_proc > 1:
                freq_disp_relation = mpi.comm.allreduce(
                    freq_disp_relation, op=mpi.MPI.MAX)

            self.deltat_dispersion_relation = (
                self.coef_deltat_dispersion_relation *
                (2. * pi / freq_disp_relation))

        # Try to compute deltat_group_vel
        try:
            freq_group, freq_phase= (
                self._compute_time_increment_group_and_phase())
        except AttributeError as e:
            print(
            '_compute_time_increment_group_and_phase is not implemented', e)
            self.deltat_group_vel = 1.
            self.deltat_phase_vel = 1.

        else:
            if mpi.nb_proc > 1:
                freq_group = mpi.comm.allreduce(
                    freq_group, op=mpi.MPI.MAX)
                freq_phase = mpi.comm.allreduce(
                    freq_phase, op=mpi.MPI.MAX)

            self.deltat_group_vel = self.coef_group / freq_group
            self.deltat_phase_vel = self.coef_phase / freq_phase

        if self.params.forcing.enable:
            self.deltat_f = self._compute_time_increment_forcing()

    def _compute_time_increment_forcing(self):
        """
        Compute time increment of the forcing.
        """
        return (1. / (self.sim.params.forcing.forcing_rate**(1./3)))

    def _compute_time_increment_group_and_phase(self):
        r"""
        Compute time increment from group velocity of the internal gravity
        waves as \omega_g = max(|c_g|) \cdot max(|k|)
        """
        N = self.params.N
        oper = self.sim.oper

        KX = oper.KX
        KZ = oper.KY
        KK_not0 = oper.KK_not0

        # Group velocity cg
        cg_kx = (N / KK_not0) * (KZ**2 / KK_not0**2)
        cg_kz = (-N / KK_not0) * ((KX / KK_not0) * (KZ / KK_not0))
        # cg = np.sqrt(cg_kx**2 + cg_kz**2)
        max_cgx = cg_kx.max()
        max_cgz = cg_kz.max()

        freq_group = (max_cgx / oper.deltax +
                      max_cgz / oper.deltay)

        # Phase velocity cp
        cp = N * (KX / KK_not0**2)
        max_cp = cp.max()

        freq_phase = max_cp / oper.deltax

        return freq_group, freq_phase

    def _compute_time_increment_CFL_uxuyb(self):
        """
        Compute time increment with the CFL condition solver ns2d.strat.
        """
        # Compute deltat_CFL at each time step.
        ux = self.sim.state.get_var('ux')
        uy = self.sim.state.get_var('uy')

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        freq_CFL = (max_ux / self.sim.oper.deltax +
                    max_uy / self.sim.oper.deltay)

        if mpi.nb_proc > 1:
            freq_CFL = mpi.comm.allreduce(freq_CFL, op=mpi.MPI.MAX)

        if freq_CFL > 0:
            deltat_CFL = self.CFL / freq_CFL
        else:
            deltat_CFL = self.deltat_max

        maybe_new_dt = min(
            deltat_CFL, self.deltat_dispersion_relation,
            self.deltat_group_vel, self.deltat_phase_vel,
            self.deltat_max)

        if self.params.forcing.enable:
            maybe_new_dt = min(maybe_new_dt, self.deltat_f)

        normalize_diff = abs(self.deltat-maybe_new_dt)/maybe_new_dt
        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt

    def one_time_step_computation(self):
        """One time step"""
        # WARNING: if the function _time_step_RK comes from an extension, its
        # execution time seems to be attributed to the function
        # one_time_step_computation by cProfile
        self._time_step_RK()
        self.sim.oper.dealiasing(self.sim.state.state_spect)

        # If no shear modes in the flow.
        if self.sim.params.NO_SHEAR_MODES:
            for ikey, key_name in enumerate(self.sim.state.state_spect.keys):
                key_fft = self.sim.state.state_spect.get_var(key_name)
                key_fft[0, :] = 0
                self.sim.state.state_spect.set_var(key_name, key_fft)

        self.sim.state.statephys_from_statespect()
        # np.isnan(np.sum seems to be really fast
        if np.isnan(np.sum(self.sim.state.state_spect[0])):
            raise ValueError(
                'nan at it = {0}, t = {1:.4f}'.format(self.it, self.t))
