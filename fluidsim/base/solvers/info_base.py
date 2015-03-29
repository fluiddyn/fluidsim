
from copy import deepcopy

from fluiddyn.util.containerxml import ContainerXML
from fluiddyn.util.util import import_class


def create_info_simul(info_solver, params):
    """Create a ContainerXML instance gathering info_solver and params."""
    info = ContainerXML(tag='info_simul')
    info.set_as_child(info_solver)
    info.set_as_child(params)
    return info


def _merged_element(el1, el2):
    result = deepcopy(el1)
    result.extend(deepcopy(el2))
    return result


class InfoSolverBase(ContainerXML):
    """Contain the information on a solver."""
    def __init__(self, **kargs):

        if 'tag' not in kargs:
            kargs['tag'] = 'solver'

        super(InfoSolverBase, self).__init__(**kargs)

        if kargs['tag'] == 'solver' and 'path_file' not in kargs:
            self._init_root()

    def _init_root(self):

        self.set_attribs({'module_name': 'fluidsim.base.solvers.base',
                          'class_name': 'SimulBase',
                          'short_name': 'Base'})

        self.set_child('classes')

        self.classes.set_child(
            'InitFields',
            attribs={'module_name': 'fluidsim.base.init_fields',
                     'class_name': 'InitFieldsBase'})

        self.classes.set_child(
            'Forcing',
            attribs={'module_name': 'fluidsim.base.forcing',
                     'class_name': 'ForcingBase'})

        self.classes.set_child(
            'Output',
            attribs={'module_name': 'fluidsim.base.output.base',
                     'class_name': 'OutputBase'})

    def import_classes(self):
        """Import the classes and return a dictionary."""
        classes = self._elemxml.findall('classes')
        dict_classes = {}
        if len(classes) == 0:
            return dict_classes
        classes = reduce(_merged_element, classes)
        for c in classes.getchildren():
            try:
                module_name = c.attrib['module_name']
                class_name = c.attrib['class_name']
            except KeyError:
                pass
            else:
                Class = import_class(module_name, class_name)
                dict_classes[c.tag] = Class

        return dict_classes

    def complete_with_classes(self):
        dict_classes = self.import_classes()
        for Class in dict_classes.values():
            if hasattr(Class, '_complete_info_solver'):
                Class._complete_info_solver(self)


