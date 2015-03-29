"""Forcing schemes (:mod:`fluidsim.base.forcing.base`)
============================================================

.. currentmodule:: fluidsim.base.forcing.base

Provides:

.. autoclass:: ForcingBase
   :members:
   :private-members:

.. autoclass:: ForcingBasePseudoSpectral
   :members:
   :private-members:

"""


class ForcingBase(object):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        info_solver.classes.Forcing.set_child('classes')

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        params.set_child(
            'forcing',
            attribs={'type': 'Random',
                     'available_types': ['Random', 'Proportional'],
                     'forcing_rate': 1,
                     'key_forced': 'rot_fft'})
        dict_classes = info_solver.classes.Forcing.import_classes()
        for Class in dict_classes.values():
            if hasattr(Class, '_complete_params_with_default'):
                try:
                    Class._complete_params_with_default(params)
                except TypeError:
                    Class._complete_params_with_default(params, info_solver)

    def __init__(self, params, sim):
        self.type_forcing = params.forcing.type

        dict_classes = sim.info.solver.classes.Forcing.import_classes()

        if self.type_forcing not in dict_classes:
            raise ValueError('Bad value for parameter forcing.type :' +
                             self.type_forcing)

        ClassForcing = dict_classes[self.type_forcing]

        self._forcing = ClassForcing(params, sim)

    def compute(self):
        self._forcing.compute()

    def get_forcing(self):
        return self._forcing.forcing_phys


class ForcingBasePseudoSpectral(ForcingBase):

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        ForcingBase._complete_params_with_default(params, info_solver)

        params.forcing.set_attribs({'nkmax_forcing': 5, 'nkmin_forcing': 4})

    def compute(self):
        self._forcing.compute()

    def get_forcing(self):
        return self._forcing.forcing_fft
