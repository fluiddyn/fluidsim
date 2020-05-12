"""Physical fields output 1d (:mod:`fluidsim.base.output.phys_fields1d`)
========================================================================

Provides:

.. autoclass:: MoviesBasePhysFields1D
   :members:
   :private-members:

.. autoclass:: PhysFieldsBase1D
   :members:
   :private-members:


"""

import numpy as np

from fluiddyn.util import mpi

from .movies import MoviesBase1D

from .phys_fields import PhysFieldsBase


class MoviesBasePhysFields1D(MoviesBase1D):
    def __init__(self, output, phys_fields):
        self.phys_fields = phys_fields
        super().__init__(output)
        self._equation = None


class PhysFieldsBase1D(PhysFieldsBase):
    def _init_movies(self):
        self.movies = MoviesBasePhysFields1D(self.output, self)

    def plot(self, field=None, time=None, numfig=None):
        is_field_ready = False

        key_field = None
        if field is None:
            key_field = self.get_key_field_to_plot(
                forbid_compute=time is not None
            )
        elif isinstance(field, np.ndarray):
            key_field = "given array"
            is_field_ready = True
        elif isinstance(field, str):
            key_field = field

        assert key_field is not None

        if time is None and not is_field_ready:
            # we have to get the field from the state
            time = self.sim.time_stepping.t
            field, _ = self.get_field_to_plot_from_state(key_field)
        else:
            self.set_of_phys_files.update_times()
            # we have to get the field from a file
            if key_field not in self.sim.state.keys_state_phys:
                raise ValueError("key not in state.keys_state_phys")

            field, time = self.get_field_to_plot(key=key_field, time=time)

        if mpi.rank == 0:
            if numfig is None:
                fig, ax = self.output.figure_axe(size_axe=None)
            else:
                fig, ax = self.output.figure_axe(numfig=numfig, size_axe=None)
            xs = self.oper.xs

            ax.plot(xs, field)
            ax.set_xlabel("x")

            self._set_title(ax, key_field, time)

    def _set_title(self, ax, key, time):
        ax.set_title(key + f", $t = {time:.3f}$\n" + self.output.summary_simul)
