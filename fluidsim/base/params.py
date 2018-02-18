"""Information on a solver (:mod:`fluidsim.base.params`)
==============================================================


Provides:

.. autoclass:: Parameters
   :members:
   :private-members:


"""

from __future__ import division, print_function

import os
from importlib import import_module
from builtins import map

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util import import_class
from fluiddyn.util import mpi

from fluidsim.base.solvers.info_base import InfoSolverBase


class Parameters(ParamContainer):
    """Contain the parameters."""
    def __ior__(self, other):
        """Defines operator `|=`

        This operator performs a union with other Parameters instances.
        In simpler words, merge missing attributes and children (members).
        Do note, this does not check for mismatch in the parameter values.

        Parameters
        ----------
        other: Parameters
            Another Parameters instance to look for missing members.

        Returns
        -------
        Parameters

        """
        if not isinstance(other, Parameters):
            raise TypeError(
                '{}. Can only merge instances of Parameters'.format(
                    type(other)))

        params1 = self
        params2 = other

        # Merge attributes
        diff_attribs = set(params2._key_attribs) - set(params1._key_attribs)

        if len(diff_attribs) > 0:
            print('Add parameter attributes: ', diff_attribs)

        for attrib in diff_attribs:
            params1._set_attrib(attrib, params2[attrib])

        # Merge childrean
        diff_children = set(params2._tag_children) - set(params1._tag_children)
        internal_attribs = [
            'attribs', 'children', 'key_attribs', 'tag', 'tag_children']

        if len(diff_children) > 0:
            print('Add parameter children: ', diff_children)

        for child in diff_children:
            child_attribs = params2[child]._make_dict()
            # Clean up intenal attributes from dictionary
            list(map(child_attribs.__delitem__, internal_attribs))

            params1._set_child(child, child_attribs)

        # Recurse
        for child in params2._tag_children:
            params1[child] |= params2[child]

        return self


def fix_old_params(params):
    """Fix old parameters with depreciated values."""
    # params.FORCING -> params.forcing.enable (2018-02-16)
    try:
        params.FORCING
    except AttributeError:
        pass
    else:
        try:
            params.forcing
        except AttributeError:
            pass
        else:
            params.forcing._set_attrib('enable', params.FORCING)


def merge_params(to_params, *other_params):
    """Merges missing parameters attributes and children of a typical
    Simulation object's parameters when compared to other parameters.
    Also, tries to replace `to_params.oper.type_fft` if found to be
    not based on FluidFFT.

    Parameters
    ----------
    to_params: Parameters

    other_params: Parameters, Parameters, ...

    """
    for other in other_params:
        to_params |= other

    def find_available_fluidfft(dim):
        """Find available FluidFFT implementations."""
        use_mpi = mpi.nb_proc > 1
        fluidfft = import_module('fluidfft.fft{}d'.format(dim))
        fluidfft_classes = (
            fluidfft.get_classes_mpi() if use_mpi else
            fluidfft.get_classes_seq())
        for k, v in fluidfft_classes.items():
            if v is not None:
                break
        else:
            raise ValueError(
                'No compatible fluidfft FFT types found for'
                '{}D, mpi={}'.format(dim, use_mpi))

        return k

    # Substitute old FFT types with newer FluidFFT implementations
    if hasattr(to_params, 'oper') and hasattr(to_params.oper, 'type_fft'):
        method = to_params.oper.type_fft
        if not (method.startswith('fft2d.') or method.startswith('fft3d.')):
            dim = 3 if hasattr(to_params.oper, 'nz') else 2
            type_fft = find_available_fluidfft(dim)
            print(
                'params.oper.type_fft', to_params.oper.type_fft, '->',
                type_fft)
            to_params.oper.type_fft = type_fft


def create_params(input_info_solver):
    """Create a Parameters instance from an InfoSolverBase instance."""
    if isinstance(input_info_solver, InfoSolverBase):
        info_solver = input_info_solver
    elif hasattr(input_info_solver, 'Simul'):
        info_solver = input_info_solver.Simul.create_default_params()
    else:
        raise ValueError('Can not create params from input input_info_solver.')

    params = Parameters(tag='params')
    dict_classes = info_solver.import_classes()

    dict_classes['Solver'] = import_class(
        info_solver.module_name, info_solver.class_name)

    for Class in list(dict_classes.values()):
        if hasattr(Class, '_complete_params_with_default'):
            try:
                Class._complete_params_with_default(params)
            except TypeError:
                try:
                    Class._complete_params_with_default(params, info_solver)
                except TypeError as e:
                    e.args += ('for class: ' + repr(Class),)
                    raise
    return params


def load_params_simul(path_dir=None):
    """Load the parameters and return a Parameters instance."""
    if path_dir is None:
        path_dir = os.getcwd()
    return Parameters(
        path_file=os.path.join(path_dir, 'params_simul.xml'))


def load_info_solver(path_dir=None):
    """Load the solver information, return an InfoSolverBase instance.

    """
    if path_dir is None:
        path_dir = os.getcwd()
    return InfoSolverBase(
        path_file=os.path.join(path_dir, 'info_solver.xml'))


# def load_info_simul(path_dir=None):
#     """Load the data and gather them in a ParamContainer instance."""

#     if path_dir is None:
#         path_dir = os.getcwd()
#     info_solver = load_info_solver(path_dir=path_dir)
#     params = load_params_simul(path_dir=path_dir)
#     info = ParamContainer(tag='info_simul')
#     info._set_as_child(info_solver)
#     info._set_as_child(params)
#     return info


if __name__ == '__main__':
    info_solver = InfoSolverBase(tag='solver')

    info_solver.complete_with_classes()

    params = create_params(info_solver)

    # info = create_info_simul(info_solver, params)
