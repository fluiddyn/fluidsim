from functools import lru_cache

from fluidsim_core.output.phys_fields import PhysFieldsABC
from fluidsim_core.output.movies import MoviesBasePhysFieldsHexa
from fluidsim_core.hexa_files import SetOfPhysFieldFiles


class PhysFields4Snek5000(PhysFieldsABC):

    _cls_movies = MoviesBasePhysFieldsHexa
    _cls_set_of_files = SetOfPhysFieldFiles

    def __init__(self, output=None):

        self.output = output
        self.params = output.params

        self.set_of_phys_files = self._cls_set_of_files(output=output)
        self.plot_hexa = self.set_of_phys_files.plot_hexa
        self.read_hexadata = self.set_of_phys_files.read_hexadata
        self.read_hexadata_from_time = (
            self.set_of_phys_files.read_hexadata_from_time
        )

        self.movies = self._cls_movies(output, self)
        self.animate = self.movies.animate
        self.interact = self.movies.interact
        self._equation = None

    def get_key_field_to_plot(self, forbid_compute=False, key_prefered=None):
        return self.set_of_phys_files.get_key_field_to_plot(key_prefered)

    def get_field_to_plot(
        self,
        key=None,
        time=None,
        idx_time=None,
        equation=None,
        interpolate_time=True,
        skip_vars=(),
    ):
        """Get the field to be plotted in process 0."""
        return self.set_of_phys_files.get_field_to_plot(
            time=time,
            idx_time=idx_time,
            key=key,
            equation=equation,
            interpolate_time=interpolate_time,
            skip_vars=skip_vars,
        )

    def get_vector_for_plot(
        self,
        from_state=False,
        time=None,
        interpolate_time=True,
        skip_vars=(),
    ):
        if from_state:
            raise ValueError("cannot get anything from the state for this solver")
        return self.set_of_phys_files.get_vector_for_plot(
            time, self._equation, skip_vars=skip_vars
        )

    @lru_cache(maxsize=None)
    def _get_axis_data(self, equation=None):
        hexa_x, _ = self.get_field_to_plot(idx_time=0, key="x", equation=equation)
        hexa_y, _ = self.get_field_to_plot(idx_time=0, key="y", equation=equation)
        return hexa_x, hexa_y
