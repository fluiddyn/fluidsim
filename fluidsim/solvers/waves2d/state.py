"""State for the Waves solver (:mod:`fluidsim.solvers.waves2d.state`)
=======================================================================
"""

from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateWaves(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver AD1D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["f_fft", "g_fft"],
                "keys_state_phys": ["f", "g"],
                "keys_computable": [],
                "keys_phys_needed": ["f", "g"],
            }
        )
