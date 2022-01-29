"""Time stepping (:mod:`fluidsim.solvers.ns2d.strat.time_stepping`)
===============================================================================


Provides:

.. autoclass:: TimeSteppingPseudoSpectralStrat
   :members:
   :private-members:

"""

from math import pi
from fluiddyn.util import mpi
from fluidsim.base.time_stepping.pseudo_spect import TimeSteppingPseudoSpectral


class TimeSteppingPseudoSpectralStrat(TimeSteppingPseudoSpectral):
    """
    Class time stepping for the solver ns2d.strat.

    """

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)

        # Add parameter coefficient CFL GROUP VELOCITY
        params.time_stepping._set_attrib("cfl_coef_group", None)

    def _init_compute_time_step(self):
        """
        Initialization compute time step solver ns2d.strat.
        """
        super()._init_compute_time_step()

        # Coefficients dt
        self.coef_deltat_dispersion_relation = 1.0

        try:
            self.coef_group = self.params.time_stepping.cfl_coef_group
        except AttributeError:
            print("self.params.time_stepping.cfl_coef_group not in params")
            self.coef_group = 1.0

        self.coef_phase = 1.0

        has_vars = self.sim.state.has_vars
        has_ux = has_vars("ux") or has_vars("vx")
        has_uy = has_vars("uy") or has_vars("vy")
        has_b = has_vars("b")

        if has_ux and has_uy and has_b:
            self.compute_time_increment_CLF = (
                self._compute_time_increment_CFL_uxuyb
            )

        # Try to compute deltat_dispersion_relation.
        try:
            self.dispersion_relation = self.sim.compute_dispersion_relation()
        except AttributeError:
            print("compute_dispersion_relation is not implemented.")
            self.deltat_dispersion_relation = 1.0
        else:
            freq_disp_relation = self.dispersion_relation.max()

            if mpi.nb_proc > 1:
                freq_disp_relation = mpi.comm.allreduce(
                    freq_disp_relation, op=mpi.MPI.MAX
                )

            self.deltat_dispersion_relation = (
                self.coef_deltat_dispersion_relation
                * (2.0 * pi / freq_disp_relation)
            )

        # Try to compute deltat_group_vel if self.coed_group is True
        if self.coef_group:
            try:
                (
                    freq_group,
                    freq_phase,
                ) = self._compute_time_increment_group_and_phase()
            except AttributeError as e:
                print(
                    "_compute_time_increment_group_and_phase is not implemented",
                    e,
                )
                self.deltat_group_vel = 1.0
                self.deltat_phase_vel = 1.0

            else:
                if mpi.nb_proc > 1:
                    freq_group = mpi.comm.allreduce(freq_group, op=mpi.MPI.MAX)
                    freq_phase = mpi.comm.allreduce(freq_phase, op=mpi.MPI.MAX)

                self.deltat_group_vel = self.coef_group / freq_group
                self.deltat_phase_vel = self.coef_phase / freq_phase

        if self.params.forcing.enable:
            self.deltat_f = self._compute_time_increment_forcing()

    def _compute_time_increment_forcing(self):
        """
        Compute time increment of the forcing.
        """
        return 1.0 / (self.sim.params.forcing.forcing_rate ** (1.0 / 3))

    def _compute_time_increment_group_and_phase(self):
        r"""
        Compute time increment from group velocity of the internal gravity
        waves as \omega_g = max(|c_g|) \cdot max(|k|)
        """
        N = self.params.N
        oper = self.sim.oper

        KX = oper.KX
        KZ = oper.KY
        K_not0 = oper.K_not0

        # Group velocity cg
        cg_kx = (N / K_not0) * (KZ**2 / K_not0**2)
        cg_kz = (-N / K_not0) * ((KX / K_not0) * (KZ / K_not0))
        # cg = np.sqrt(cg_kx**2 + cg_kz**2)
        max_cgx = cg_kx.max()
        max_cgz = cg_kz.max()

        freq_group = max_cgx / oper.deltax + max_cgz / oper.deltay

        # Phase velocity cp
        cp = N * (KX / K_not0**2)
        max_cp = cp.max()

        freq_phase = max_cp / oper.deltax

        return freq_group, freq_phase

    def _compute_time_increment_CFL_uxuyb(self):
        """
        Compute time increment with the CFL condition solver ns2d.strat.
        """
        # Compute deltat_CFL at each time step.
        ux = self.sim.state.get_var("ux")
        uy = self.sim.state.get_var("uy")

        max_ux = abs(ux).max()
        max_uy = abs(uy).max()
        freq_CFL = max_ux / self.sim.oper.deltax + max_uy / self.sim.oper.deltay

        if mpi.nb_proc > 1:
            freq_CFL = mpi.comm.allreduce(freq_CFL, op=mpi.MPI.MAX)

        if freq_CFL > 0:
            deltat_CFL = self.CFL / freq_CFL
        else:
            deltat_CFL = self.deltat_max

        # Removed phase velocity (considered not relevant)
        if not self.coef_group:
            maybe_new_dt = min(
                deltat_CFL, self.deltat_dispersion_relation, self.deltat_max
            )
        else:
            maybe_new_dt = min(
                deltat_CFL,
                self.deltat_dispersion_relation,
                self.deltat_group_vel,
                self.deltat_max,
            )

        if self.params.forcing.enable:
            maybe_new_dt = min(maybe_new_dt, self.deltat_f)

        normalize_diff = abs(self.deltat - maybe_new_dt) / maybe_new_dt
        if normalize_diff > 0.02:
            self.deltat = maybe_new_dt
