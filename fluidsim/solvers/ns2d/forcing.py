

from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import Proportional

from fluidsim.base.forcing.specific import \
    TimeCorrelatedRandomPseudoSpectral as Random


class ForcingNS2D(ForcingBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        ForcingBasePseudoSpectral._complete_info_solver(info_solver)
        classes = info_solver.classes.Forcing.classes

        package = 'fluidsim.solvers.ns2d.forcing'

        classes.set_child(
            'Random',
            attribs={'module_name': package,
                     'class_name': 'Random'})

        classes.set_child(
            'Proportional',
            attribs={'module_name': package,
                     'class_name': 'Proportional'})
