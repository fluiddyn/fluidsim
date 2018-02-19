"""State boussinesq NS3D solver (:mod:`fluidsim.solvers.ns3d.bouss.state`)
==========================================================================
"""

from ..state import StateNS3D


class StateNS3DBouss(StateNS3D):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver NS3D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        keys_state_phys = ['vx', 'vy', 'vz', 'b']

        info_solver.classes.State._set_attribs({
            'keys_state_spect': [k + '_fft' for k in keys_state_phys],
            'keys_state_phys': keys_state_phys,
            'keys_phys_needed': keys_state_phys,
            'keys_computable': [],
            'keys_linear_eigenmodes': ['rot_fft']
        })
