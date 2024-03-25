"""Utility to produce a dataframe from a set of simulations

"""

import json
import hashlib
import inspect
from abc import ABC, abstractmethod

from pathlib import Path

from pandas import DataFrame

from rich.progress import track


class DataframeMaker(ABC):
    """To produce a Pandas dataframe from a set of simulations"""

    @abstractmethod
    def get_time_start_from_path(self, path):
        """ "Get first time"""
        # t_start, _ = times_start_last_from_path(path)
        # return t_start

    @abstractmethod
    def get_time_last_from_path(self, path):
        """ "Get last saved time"""
        # return get_last_time_spatial_means_from_path(path)

    @abstractmethod
    def load_sim(self, path):
        """Load a simulation object"""
        # return load_sim_for_plot(path, hide_stdout=True)

    def get_mean_values_from_path(
        self, path, tmin=None, tmax=None, use_cache=True, customize=None
    ):
        """Get a dict of scalar values characterizing the simulation

        Parameters
        ----------

        tmin: float
            Minimum time

        tmax: float
            Maximum time

        use_cache: bool
            If True, return the cached result

        customize: callable

            If not None, called as ``customize(result, self.sim)`` to modify the
            returned dict.

        Examples
        --------

        .. code-block:: python

            def customize(result, sim):
                result["Rb"] = float(sim.params.short_name_type_run.split("_Rb")[-1])
            get_mean_values_from_path(path, customize=customize)

        """

        if (
            tmin is None
            or isinstance(tmin, str)
            or tmax is None
            or isinstance(tmax, str)
        ):
            t_start = self.get_time_start_from_path(path)
            t_last = self.get_time_last_from_path(path)

        if tmin is None:
            tmin = t_start
        elif isinstance(tmin, str):
            if tmin.startswith("t_start+"):
                tmin = t_start + float(tmin.split("t_start+")[-1])
            elif tmin.startswith("t_last-"):
                tmin = t_last - float(tmin.split("t_last-")[-1])
            else:
                raise ValueError(
                    f"isinstance(tmin, str) and {tmin=} but tmin has to start by "
                    '"t_start+" or "t_last-"'
                )
        tmin = float(tmin)

        if tmax is None:
            tmax = t_last
        elif isinstance(tmax, str):
            if tmax.startswith("t_start+"):
                tmax = t_start + float(tmax.split("t_start+")[-1])
            elif tmax.startswith("t_last-"):
                tmax = t_last - float(tmax.split("t_last-")[-1])
            else:
                raise ValueError(
                    f"isinstance(tmax, str) and {tmax=} but tmin has to start by "
                    '"t_start+" or "t_last-"'
                )
        tmax = float(tmax)

        cache_dir = Path(path) / ".cache"
        cache_dir.mkdir(exist_ok=True)

        if customize is not None:
            source = inspect.getsource(customize).encode().strip()
            hash = hashlib.sha256(source).hexdigest()[:16]
            part_customize = f"_customize{hash}"
        else:
            part_customize = ""

        cache_file = cache_dir / (
            f"mean_values_tmin{tmin}_tmax{tmax}{part_customize}.json"
        )

        if use_cache and cache_file.exists():
            with open(cache_file, "r") as file:
                return json.load(file)

        sim = self.load_sim(path)

        result = sim.output._compute_mean_values(tmin, tmax)
        if customize is not None:
            customize(result, sim)

        print("saving", cache_file)
        with open(cache_file, "w") as file:
            json.dump(result, file, indent=2)
        return result

    def get_dataframe_from_paths(
        self, paths, tmin=None, tmax=None, use_cache=True, customize=None
    ):
        """Produce a dataframe from a set of simulations.

        Uses ``sim.output.get_mean_values``

        """
        values = []
        for path in track(paths, "Getting the mean values"):
            values.append(
                self.get_mean_values_from_path(
                    path, tmin, tmax, use_cache, customize
                )
            )
        return DataFrame(values)
