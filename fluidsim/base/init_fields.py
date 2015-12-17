"""Initialisation of the fields (:mod:`fluidsim.base.init_fields`)
========================================================================

.. currentmodule:: fluidsim.base.init_fields

Provides:

.. autoclass:: InitFieldsBase
   :members:
   :private-members:

"""

import h5py

from copy import deepcopy

from fluiddyn.util import mpi

from fluidsim.base.setofvariables import SetOfVariables


class InitFieldsBase(object):
    """Initialization of the fields (base class)."""

    @staticmethod
    def _complete_info_solver(info_solver, classes=None):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.InitFields._set_child('classes')

        classesXML = info_solver.classes.InitFields.classes

        if classes is None:
            classes = []

        classes.extend([InitFieldsFromFile, InitFieldsFromSimul,
                        InitFieldsManual, InitFieldsConstant])

        for cls in classes:
            classesXML._set_child(
                cls.tag,
                attribs={'module_name': cls.__module__,
                         'class_name': cls.__name__})

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        params._set_child('init_fields', attribs={
            'type': 'constant',
            'available_types': []})

        dict_classes = info_solver.classes.InitFields.import_classes()

        for Class in dict_classes.values():
            if hasattr(Class, '_complete_params_with_default'):
                try:
                    Class._complete_params_with_default(params)
                except TypeError:
                    try:
                        Class._complete_params_with_default(
                            params, info_solver)
                    except TypeError as e:
                        e.args += ('for class: ' + repr(Class),)
                        raise

    def __init__(self, sim):

        self.sim = sim
        params = sim.params
        oper = sim.oper

        self.params = params
        self.oper = oper

        type_init = params.init_fields.type
        if type_init not in params.init_fields.available_types:
            raise ValueError(type_init +
                             ' is not an available flow initialization.')

        dict_classes = sim.info.solver.classes.InitFields.import_classes()
        Class = dict_classes[type_init]
        self._specific_init_fields = Class(sim)

    def __call__(self):
        self.sim.state.is_initialized = True
        self._specific_init_fields()


class SpecificInitFields(object):
    tag = 'specific'

    @classmethod
    def _complete_params_with_default(cls, params):
        params.init_fields.available_types.append(cls.tag)

    def __init__(self, sim):
        self.sim = sim


class InitFieldsFromFile(SpecificInitFields):

    tag = 'from_file'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsFromFile, cls)._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={'path': ''})

    def __call__(self):

        # Warning: this function is for 2d pseudo-spectral solver!
        # We have to write something more general.

        params = self.sim.params

        path_file = params.init_fields.from_file.path

        if mpi.rank == 0:
            try:
                f = h5py.File(path_file, 'r')
            except:
                raise ValueError('file '+path_file+' is really a hd5 file?')

            print ('Load state from file:\n[...]'+path_file[-75:])

            try:
                group_oper = f['/info_simul/params/oper']
            except:
                raise ValueError(
                    'file '+path_file+' does not contain a params object')

            try:
                group_state_phys = f['/state_phys']
            except:
                raise ValueError('file ' + path_file +
                                 ' does not contain a state_phys object')

            nx_file = group_oper.attrs['nx']
            ny_file = group_oper.attrs['ny']
            Lx_file = group_oper.attrs['Lx']
            Ly_file = group_oper.attrs['Ly']

            if isinstance(nx_file, list):
                nx_file = nx_file.item()
                ny_file = ny_file.item()
                Lx_file = Lx_file.item()
                Ly_file = Ly_file.item()

            if params.oper.nx != nx_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.nx != params_file.nx')

            if params.oper.ny != ny_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.ny != params_file.ny')

            if params.oper.Lx != Lx_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.params.oper.Lx != params_file.Lx')

            if params.oper.Ly != Ly_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.params.oper.Ly != params_file.Ly')

            keys_state_phys_file = group_state_phys.keys()
        else:
            keys_state_phys_file = {}

        if mpi.nb_proc > 1:
            keys_state_phys_file = mpi.comm.bcast(keys_state_phys_file)

        state_phys = self.sim.state.state_phys
        keys_phys_needed = self.sim.info.solver.classes.State.keys_phys_needed
        for k in keys_phys_needed:
            if k in keys_state_phys_file:
                if mpi.rank == 0:
                    field_seq = group_state_phys[k][...]
                else:
                    field_seq = self.sim.oper.constant_arrayX()

                if mpi.nb_proc > 1:
                    field_loc = self.sim.oper.scatter_Xspace(field_seq)
                else:
                    field_loc = field_seq
                state_phys.set_var(k, field_loc)
            else:
                state_phys.set_var(k, self.sim.oper.constant_arrayX(value=0.))

        if mpi.rank == 0:
            time = group_state_phys.attrs['time']

            try:
                it = group_state_phys.attrs['it']
            except KeyError:
                # compatibility with older versions
                it = 0
            f.close()
        else:
            time = 0.
            it = 0

        if mpi.nb_proc > 1:
            time = mpi.comm.bcast(time)
            it = mpi.comm.bcast(it)

        self.sim.state.statefft_from_statephys()
        self.sim.state.statephys_from_statefft()
        self.sim.time_stepping.t = time
        self.sim.time_stepping.it = it


class InitFieldsFromSimul(SpecificInitFields):

    tag = 'from_simul'

    def __call__(self):
        self.sim.init_fields.get_state_from_simul = self._get_state_from_simul

    def _get_state_from_simul(self, sim_in):

        # Warning: this function is for 2d pseudo-spectral solver!
        # We have to write something more general.
        # It should be done directly in the operators.

        if mpi.nb_proc > 1:
            raise ValueError('BE CARREFUL, THIS WILL BE WRONG !'
                             '  DO NOT USE THIS METHOD WITH MPI')

        self.sim.time_stepping.t = sim_in.time_stepping.t

        if (self.params.oper.nx == sim_in.params.oper.nx and
                self.params.oper.ny == sim_in.params.oper.ny):
            state_fft = deepcopy(sim_in.state.state_fft)
        else:
            # modify resolution
            # state_fft = SetOfVariables('state_fft')
            state_fft = SetOfVariables(like=self.sim.state.state_fft)
            keys_state_fft = sim_in.info.solver.classes.State['keys_state_fft']
            for k in keys_state_fft:
                field_fft_seq_in = sim_in.state.state_fft[k]
                field_fft_seq_new_res = \
                    self.sim.oper.constant_arrayK(value=0.)
                [nk0_seq, nk1_seq] = field_fft_seq_new_res.shape
                [nk0_seq_in, nk1_seq_in] = field_fft_seq_in.shape

                nk0_min = min(nk0_seq, nk0_seq_in)
                nk1_min = min(nk1_seq, nk1_seq_in)

                # it is a little bit complicate to take into account ky
                for ik1 in xrange(nk1_min):
                    field_fft_seq_new_res[0, ik1] = field_fft_seq_in[0, ik1]
                    field_fft_seq_new_res[nk0_min/2, ik1] = \
                        field_fft_seq_in[nk0_min/2, ik1]
                for ik0 in xrange(1, nk0_min/2):
                    for ik1 in xrange(nk1_min):
                        field_fft_seq_new_res[ik0, ik1] = \
                            field_fft_seq_in[ik0, ik1]
                        field_fft_seq_new_res[-ik0, ik1] = \
                            field_fft_seq_in[-ik0, ik1]

                state_fft[k] = field_fft_seq_new_res

        if self.sim.output.name_solver == sim_in.output.name_solver:
            self.sim.state.state_fft = state_fft
        else:  # complicated case... untested solution !
            # state_fft = SetOfVariables('state_fft')
            raise ValueError('Not yet implemented...')
            for k in self.sim.info.solver.classes.State['keys_state_fft']:
                if k in sim_in.info.solver.classes.State['keys_state_fft']:
                    self.sim.state.state_fft[k] = state_fft[k]
                else:
                    self.sim.state.state_fft[k] = \
                        self.oper.constant_arrayK(value=0.)

        self.sim.state.statephys_from_statefft()


class InitFieldsManual(SpecificInitFields):

    tag = 'manual'

    def __call__(self):
        self.sim.state.is_initialized = False
        self.sim.output.print_stdout(
            'Manual initialization of the fields in selected. '
            'Do not forget to initialize them.')


class InitFieldsConstant(SpecificInitFields):

    tag = 'constant'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsConstant, cls)._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={'value': 1.})

    def __call__(self):
        value = self.sim.params.init_fields.constant.value
        self.sim.state.state_phys.initialize(value)
        self.sim.state.statefft_from_statephys()

#     def init_fields_wave(self):
#         ikx = self.sim.params.ikx
#         eta0 = self.sim.params.eta0

#         # BE CARREFUL, THIS WON'T WORK WITH MPI !!!
#         if mpi.rank == 0:
#             print 'init_fields_wave(ikx = {0:4d}, eta0 = {1:7.2e})'.format(
#                 ikx, eta0)
#             print 'kx[ikx] = {0:8.2f}'.format(self.oper.kxE[ikx])

#         if mpi.nb_proc > 1:
#             raise ValueError('BE CARREFUL, THIS WILL BE WRONG !'
#                              '  DO NOT USE THIS METHOD WITH MPI '
#                              '(or rewrite it :-)')

#         eta_fft = self.oper.constant_arrayK(value=0.)
#         ux_fft = self.oper.constant_arrayK(value=0.)
#         uy_fft = self.oper.constant_arrayK(value=0.)

#         eta_fft[0, self.sim.params.ikx] = 0.1*eta0
#         # eta_fft[ikx, 0] = 0.1j*eta0

#         self.oper.project_fft_on_realX(eta_fft)

# #        ux_fft[0,ikx] = 1.j*eta0
# #        uy_fft[0,ikx] = 1.j*eta0

#         div_fft = self.oper.constant_arrayK(value=0.)
#         div_fft[ikx, 0] = eta0
#         div_fft[0, ikx] = eta0
#         self.oper.project_fft_on_realX(div_fft)
#         ux_fft, uy_fft = self.oper.vecfft_from_divfft(div_fft)

#         return eta_fft, ux_fft, uy_fft

