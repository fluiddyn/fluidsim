"""Abstract base classes for Physical fields output

These classes can be used to define classes working together with
:class:`fluidsim_core.output.movies.MoviesBasePhysFields`.

.. autoclass:: PhysFieldsABC
   :members:
   :private-members:
   :undoc-members:

.. autoclass:: SetOfPhysFieldFilesABC
   :members:
   :private-members:
   :undoc-members:

"""

from typing import Optional

from abc import ABCMeta, abstractmethod


class SetOfPhysFieldFilesABC(metaclass=ABCMeta):
    times: list

    @abstractmethod
    def update_times(self):
        ...

    @abstractmethod
    def get_min_time(self):
        ...


class PhysFieldsABC(metaclass=ABCMeta):
    _equation: Optional[str]
    _can_plot_quiver: Optional[bool]
    set_of_phys_files: SetOfPhysFieldFilesABC

    @abstractmethod
    def get_key_field_to_plot(self, forbid_compute=False, key_field_to_plot=None):
        ...

    @abstractmethod
    def get_field_to_plot(
        self,
        key=None,
        time=None,
        idx_time=None,
        equation=None,
        interpolate_time=True,
    ):
        """Get the field to be plotted in process 0."""
        ...

    @abstractmethod
    def get_vector_for_plot(self):
        ...

    @abstractmethod
    def _quiver_plot(self, ax, vecx="ux", vecy="uy", XX=None, YY=None, skip=None):
        """Superimposes a quiver plot of velocity vectors with a given axis
        object corresponding to a 2D contour plot.

        """
        ...

    @abstractmethod
    def _set_title(self, ax, key, time, vmax=None):
        ...

    @abstractmethod
    def _get_axis_data(equation=None):
        ...
