"""Output (:mod:`fluidsim.solvers.ns2d.strat.output`)
===============================================

Provides the modules:

.. autosummary::
   :toctree:

   print_stdout
   spatial_means
   spectra
   spect_energy_budget

and the main output class for the ns2d solver:

.. autoclass:: OutputStrat
   :members:
   :private-members:

"""

import numpy as np

from fluidsim.base.output import OutputBasePseudoSpectral

from fluidsim.solvers.ns2d.output import Output

class OutputStrat(Output):
    """Output for ns2d.strat solver."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        Output._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = 'fluidsim.solvers.ns2d.strat.output'

        classes.PrintStdOut.module_name = base_name_mod + '.print_stdout'
        classes.PrintStdOut.class_name = 'PrintStdOutNS2DStrat'

        classes.PhysFields.class_name = 'PhysFieldsBase2D'

        attribs={
            'module_name': base_name_mod + '.spectra',
            'class_name': 'SpectraNS2DStrat'}
        classes.Spectra._set_attribs(attribs)

        # classes._set_child(
        #    'Spectra',
        #    attribs={'module_name': base_name_mod + '.spectra',
        #             'class_name': 'SpectraNS2DStrat'})

        attribs={
            'module_name': base_name_mod + '.spatial_means',
            'class_name': 'SpatialMeansNS2DStrat'}
        classes.spatial_means._set_attribs(attribs)

        # classes._set_child(
        #     'spatial_means',
        #     attribs={'module_name': base_name_mod + '.spatial_means',
        #              'class_name': 'SpatialMeansNS2DStrat'})
        attribs = {
            'module_name': base_name_mod + '.spect_energy_budget',
            'class_name': 'SpectralEnergyBudgetNS2DStrat'}
        classes.spect_energy_budg._set_attribs(attribs)

        # attribs = {
        #     'module_name': base_name_mod + '.spect_energy_budget',
        #     'class_name': 'SpectralEnergyBudgetNS2DStrat'}
        # classes._set_child('spect_energy_budg', attribs=attribs)

        # attribs = {
        #     'module_name': 'fluidsim.base.output.increments',
        #     'class_name': 'Increments'}
        # classes._set_child('increments', attribs=attribs)
