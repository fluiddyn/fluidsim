
"""InitFieldsSW1L"""

from fluidsim.base.init_fields import InitFieldsBase

from fluidsim.solvers.ns2d.init_fields import (
    InitFieldsNoise as InitFieldsNoiseNS2D)

from fluidsim.solvers.ns2d.init_fields import InitFieldsJet, InitFieldsDipole


class InitFieldsNoise(InitFieldsNoiseNS2D):

    def __call__(self):
        rot_fft, ux_fft, uy_fft = self.compute_rotuxuy_fft()
        self.sim.state.init_from_uxuyfft(ux_fft, uy_fft)


class InitFieldsSW1L(InitFieldsBase):
    """Init the fields for the solver SW1L."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver."""

        InitFieldsBase._complete_info_solver(info_solver)

        classesXML = info_solver.classes.InitFields.classes

        classes = [InitFieldsNoise, InitFieldsJet, InitFieldsDipole]

        for cls in classes:
            classesXML.set_child(
                cls.tag,
                attribs={'module_name': cls.__module__,
                         'class_name': cls.__name__})
