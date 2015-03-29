
"""InitFieldsNS2D"""

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase



class InitFieldsNS2D(InitFieldsBase):
    """Init the fields for the solver NS2D."""

    implemented_flows = ['NOISE', 'CONSTANT', 'LOAD_FILE', 'DIPOLE', 'JET']

    def __call__(self):
        """Init the state (in physical and Fourier space) and time"""
        sim = self.sim

        type_flow_init = self.get_and_check_type_flow_init()

        if type_flow_init == 'DIPOLE':
            rot_fft, ux_fft, uy_fft = self.init_fields_1dipole()
            tasks_complete_init = ['Fourier_to_phys']
        elif type_flow_init == 'JET':
            rot_fft, ux_fft, uy_fft = self.init_fields_jet()
            tasks_complete_init = ['Fourier_to_phys']
        elif type_flow_init == 'NOISE':
            rot_fft, ux_fft, uy_fft = self.init_fields_noise()
            tasks_complete_init = ['Fourier_to_phys']
        elif type_flow_init == 'LOAD_FILE':
            self.get_state_from_file(self.params.init_fields.path_file)
            tasks_complete_init = []
        elif type_flow_init == 'CONSTANT':
            rot_fft = sim.oper.constant_arrayK(value=0.)
            if mpi.rank == 0:
                rot_fft[1, 0] = 1.
            tasks_complete_init = ['Fourier_to_phys']
        else:
            raise ValueError('bad value of params.type_flow_init')

        if 'Fourier_to_phys' in tasks_complete_init:
            sim.oper.dealiasing(rot_fft)
            sim.state.state_fft['rot_fft'] = rot_fft

            sim.state.statephys_from_statefft()
