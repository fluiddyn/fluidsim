"""Spatial means regions
========================

.. autoclass:: SpatialMeansRegions
   :members:
   :private-members:

"""

import numbers

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.output.base import SpecificOutput

from . import SimulExtender


class SpatialMeansRegions(SimulExtender, SpecificOutput):
    """Specific output for the MILESTONE simulations

    It is still a work in progress.

    """

    _tag = "spatial_means_regions"
    _module_name = "fluidsim.extend_simul.spatial_means_regions_milestone"

    def __init__(self, output):

        params = output.sim.params

        params_cls = params.output.spatial_means_regions
        self.xmin_given = params_cls.xmin
        self.xmax_given = params_cls.xmax

        if isinstance(self.xmin_given, numbers.Number):
            self.xmin_given = [self.xmin_given]

        if isinstance(self.xmax_given, numbers.Number):
            self.xmax_given = [self.xmax_given]

        nb_regions = len(self.xmin_given)

        if len(self.xmax_given) != nb_regions:
            raise ValueError("len(self.xmax_given) != len(self.xmin_given)")

        oper = output.sim.oper

        Lx = params.oper.Lx

        self.info_regions = []

        _, _, ix_seq_start = oper.seq_indices_first_X
        nx_loc = oper.shapeX_loc[2]

        for xmin, xmax in zip(self.xmin_given, self.xmax_given):

            xmin, xmax = Lx * xmin, Lx * xmax

            ixmin = np.argmin(abs(oper.x_seq - xmin))
            xmin = oper.x_seq[ixmin]

            ixmin_loc = ixmin - ix_seq_start
            if ixmin_loc < 0 or ixmin_loc > nx_loc - 1:
                # this limit is not in this process
                ixmin_loc = None

            ixmax = np.argmin(abs(oper.x_seq - xmax))
            xmax = oper.x_seq[ixmax]
            ixmax_loc = ixmax - ix_seq_start
            ixstop_loc = ixmax_loc + 1
            if ixmax_loc < 0 or ixmax_loc > nx_loc - 1:
                # this limit is not in this process
                ixmax_loc = None

            mask_loc = np.zeros(shape=oper.shapeX_loc, dtype=np.int8)
            mask_loc[:, :, ixmin_loc:ixstop_loc] = 1

            self.info_regions.append(
                (xmin, xmax, ixmin, ixmax, ixmin_loc, ixmax_loc, mask_loc)
            )

        super().__init__(
            output,
            period_save=params.output.periods_save.spatial_means_regions,
        )

        if self.period_save != 0:
            self._save_one_time()


    def _init_files(self, arrays_1st_time=None):

        if mpi.rank == 0:
            pass

    @classmethod
    def complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)
        params.output._set_child(cls._tag, attribs={"xmin": 0.25, "xmax": 0.75})

    @classmethod
    def get_modif_info_solver(cls):
        """Create a function to modify ``info_solver``.

        Note that this function is called when the object ``info_solver`` has
        not yet been created (and cannot yet be modified)! This is why one
        needs to create a function that will be called later to modify
        ``info_solver``.

        """

        def modif_info_solver(info_solver):
            info_solver.classes.Output.classes._set_child(
                "SpatialMeansRegions",
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    def _online_save(self):
        if self._has_to_online_save():
            self._save_one_time()

    def _save_one_time(self):
        tsim = self.sim.time_stepping.t
        self.t_last_save = tsim
        return NotImplemented

    def load(self):
        return NotImplemented

    def plot(self):
        return NotImplemented
