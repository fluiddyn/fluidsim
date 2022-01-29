"""State stratified NS3D solver (:mod:`fluidsim.solvers.ns3d.strat.state`)
==========================================================================
"""

from fluidsim.solvers.ns3d.state import StateNS3D


class StateNS3DStrat(StateNS3D):
    """State for the solver ns3d.strat.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    """

    @classmethod
    def _complete_info_solver(cls, info_solver):
        """Complete the ParamContainer info_solver."""

        super()._complete_info_solver(info_solver)

        info_State = info_solver.classes.State
        keys_state_phys = ["vx", "vy", "vz", "b"]
        info_State.keys_state_spect = [k + "_fft" for k in keys_state_phys]
        info_State.keys_state_phys = keys_state_phys
        info_State.keys_phys_needed = keys_state_phys

    def init_from_vxvyvzfft(self, vx_fft, vy_fft, vz_fft):
        self.state_spect.set_var("vx_fft", vx_fft)
        self.state_spect.set_var("vy_fft", vy_fft)
        self.state_spect.set_var("vz_fft", vz_fft)
        self.state_spect.set_var("b_fft", 0.0)

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def compute_energy_phys(self):
        vx = self.state_phys.get_var("vx")
        vy = self.state_phys.get_var("vy")
        vz = self.state_phys.get_var("vz")
        E_K = 0.5 * (vx**2 + vy**2 + vz**2)
        E_A = 0.5 / self.sim.params.N**2 * self.state_phys.get_var("b") ** 2
        return self.sim.oper.mean_space(E_K + E_A)
