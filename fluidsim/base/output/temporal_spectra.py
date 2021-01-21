"""
FrequencySpectra (:mod:`fluidsim.solvers.ns3d.output.temporal_spectra`)
==============================================================================


Provides:

.. autoclass:: TemporalSpectra
   :members:
   :private-members:

"""

import sys
import time
from pathlib import Path
import glob

import numpy as np
from scipy import signal
import h5py

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput


class TemporalSpectra(SpecificOutput):
    """
    Computes the temporal spectra.
    """

    _tag = "temporal_spectra"
    # _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "temporal_spectra"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(
            tag,
            attribs={
                "HAS_TO_PLOT_SAVED": False,
                "probes_deltax": 0.1,  # m
                "probes_deltay": 0.1,  # m
                "probes_deltaz": 0.1,  # m
                "probes_region": (0, 1, 0, 1, 0, 1),  # m
                "file_max_size": 10.0,  # MB
            },
        )

    def __init__(self, output):
        params = output.sim.params
        params_tspec = params.output.temporal_spectra
        super().__init__(
            output,
            period_save=params.output.periods_save.temporal_spectra,
            has_to_plot_saved=params_tspec.HAS_TO_PLOT_SAVED,
        )

        self.t_last_save = -self.period_save

        # Parameters
        self.probes_deltax = params_tspec.probes_deltax
        self.probes_deltay = params_tspec.probes_deltay
        self.probes_deltaz = params_tspec.probes_deltaz
        self.period_save = params.output.periods_save.temporal_spectra

        self.probes_region = params_tspec.probes_region
        (
            probes_xmin,
            probes_xmax,
            probes_ymin,
            probes_ymax,
            probes_zmin,
            probes_zmax,
        ) = self.probes_region

        self.file_max_size = params_tspec.file_max_size

        self.keys_fields = self.sim.info_solver.classes.State.keys_state_phys

        oper = self.sim.oper
        X, Y, Z = oper.get_XYZ_loc()

        # round probes positions to gridpoints
        # probes spacing should at least be oper grid spacing
        self.probes_deltax = max(
            oper.deltax, oper.deltax * round(self.probes_deltax / oper.deltax)
        )
        probes_xmin = oper.deltax * round(probes_xmin / oper.deltax)

        self.probes_deltay = max(
            oper.deltay, oper.deltay * round(self.probes_deltay / oper.deltay)
        )
        probes_ymin = oper.deltay * round(probes_ymin / oper.deltay)

        self.probes_deltaz = max(
            oper.deltaz, oper.deltaz * round(self.probes_deltaz / oper.deltaz)
        )
        probes_zmin = oper.deltaz * round(probes_zmin / oper.deltaz)

        # make sure probes region is not empty
        probes_xmax = max(probes_xmax, probes_xmin + 1e-15)
        probes_ymax = max(probes_ymax, probes_ymin + 1e-15)
        probes_zmax = max(probes_zmax, probes_zmin + 1e-15)

        # global probes coordinates
        self.probes_x_seq = np.arange(
            probes_xmin, probes_xmax, self.probes_deltax
        )
        self.probes_y_seq = np.arange(
            probes_ymin, probes_ymax, self.probes_deltay
        )
        self.probes_z_seq = np.arange(
            probes_zmin, probes_zmax, self.probes_deltaz
        )

        probes_nb_seq = (
            self.probes_x_seq.size
            * self.probes_y_seq.size
            * self.probes_z_seq.size
        )

        # data directory
        dir_name = "probes"
        self.path_dir = Path(self.sim.output.path_run) / dir_name
        if not output._has_to_save:
            self.period_save = 0.0
        if self.period_save == 0.0:
            return
        else:
            if mpi.rank == 0:
                self.path_dir.mkdir(exist_ok=True)

        # check for existing files
        files = list(self.path_dir.glob("rank*"))
        if files:
            # check values in files
            with h5py.File(files[0], "r") as file:
                if file["nb_proc"][()] != mpi.nb_proc:
                    raise ValueError("process number is different from files")
                if not (
                    np.allclose(file["probes_x_seq"][:], self.probes_x_seq)
                    and np.allclose(file["probes_y_seq"][:], self.probes_y_seq)
                    and np.allclose(file["probes_z_seq"][:], self.probes_z_seq)
                ):
                    raise ValueError("probes position are different from files")
            # init from files
            files = [f for f in files if f.name.startswith(f"rank{mpi.rank:04}")]
            if files:
                self.path_file = files[-1]
                self.file_nb = int(self.path_file.name[13:17])
                with h5py.File(self.path_file) as file:
                    self.probes_x_loc = file["probes_x_loc"][:]
                    self.probes_y_loc = file["probes_y_loc"][:]
                    self.probes_z_loc = file["probes_z_loc"][:]
                    self.probes_ix_loc = file["probes_ix_loc"][:]
                    self.probes_iy_loc = file["probes_iy_loc"][:]
                    self.probes_iz_loc = file["probes_iz_loc"][:]
                    self.probes_nb_loc = self.probes_x_loc.size
                    self.file_write_nb = file["times"].size
            else:
                # no probes in proc
                self.path_file = None
                self.file_nb = 0
                self.file_write_nb = 0
                self.probes_nb_loc = 0
                self.probes_x_loc = []
                self.probes_y_loc = []
                self.probes_z_loc = []
                self.probes_ix_loc = []
                self.probes_iy_loc = []
                self.probes_iz_loc = []

            # files max size (float64 = 8e-6 MB)
            probes_init_size = 3 * (2 * self.probes_nb_loc + probes_nb_seq) * 8e-6
            probes_write_size = (4 * self.probes_nb_loc + 1) * 8e-6
            self.file_max_write = int(
                (self.file_max_size - probes_init_size) / probes_write_size
            )
        else:
            # no files were found : initialize from params
            # local probes coordinates
            cond = (self.probes_x_seq >= X.min()) & (self.probes_x_seq <= X.max())
            self.probes_x_loc = self.probes_x_seq[cond]
            cond = (self.probes_y_seq >= Y.min()) & (self.probes_y_seq <= Y.max())
            self.probes_y_loc = self.probes_y_seq[cond]
            cond = (self.probes_z_seq >= Z.min()) & (self.probes_z_seq <= Z.max())
            self.probes_z_loc = self.probes_z_seq[cond]

            self.probes_nb_loc = (
                self.probes_x_loc.size
                * self.probes_y_loc.size
                * self.probes_z_loc.size
            )

            # local probes indices
            self.probes_ix_loc = np.empty(self.probes_nb_loc, dtype=int)
            self.probes_iy_loc = np.empty_like(self.probes_ix_loc)
            self.probes_iz_loc = np.empty_like(self.probes_ix_loc)
            probe_i = 0
            for probe_x in self.probes_x_loc:
                for probe_y in self.probes_y_loc:
                    for probe_z in self.probes_z_loc:
                        probe_iz, probe_iy, probe_ix = np.where(
                            (abs(X - probe_x) <= oper.deltax / 2)
                            & (abs(Y - probe_y) <= oper.deltay / 2)
                            & (abs(Z - probe_z) <= oper.deltaz / 2)
                        )
                        self.probes_ix_loc[probe_i] = probe_ix
                        self.probes_iy_loc[probe_i] = probe_iy
                        self.probes_iz_loc[probe_i] = probe_iz
                        probe_i += 1
            self.probes_x_loc = X[
                self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
            ]
            self.probes_y_loc = Y[
                self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
            ]
            self.probes_z_loc = Z[
                self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
            ]

            # files max size (float64 = 8e-6 MB)
            probes_init_size = 3 * (2 * self.probes_nb_loc + probes_nb_seq) * 8e-6
            probes_write_size = (4 * self.probes_nb_loc + 1) * 8e-6
            self.file_max_write = int(
                (self.file_max_size - probes_init_size) / probes_write_size
            )

            # initialize files
            self.file_nb = 0
            self.file_write_nb = 0
            if self.probes_nb_loc > 0:
                self._init_new_file()

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self):
        """Initializes a new file"""
        self.path_file = (
            self.path_dir / f"rank{mpi.rank:04}_file{self.file_nb:04}.hdf5"
        )
        with h5py.File(self.path_file, "w") as file:
            create_ds = file.create_dataset
            create_ds("nb_proc", data=mpi.nb_proc)
            create_ds("probes_x_seq", data=self.probes_x_seq)
            create_ds("probes_y_seq", data=self.probes_y_seq)
            create_ds("probes_z_seq", data=self.probes_z_seq)
            create_ds("probes_x_loc", data=self.probes_x_loc)
            create_ds("probes_y_loc", data=self.probes_y_loc)
            create_ds("probes_z_loc", data=self.probes_z_loc)
            create_ds("probes_ix_loc", data=self.probes_ix_loc)
            create_ds("probes_iy_loc", data=self.probes_iy_loc)
            create_ds("probes_iz_loc", data=self.probes_iz_loc)
            create_ds(
                "probes_vx_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, self.file_max_write),
            )
            create_ds(
                "probes_vy_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, self.file_max_write),
            )
            create_ds(
                "probes_vz_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, self.file_max_write),
            )
            create_ds(
                "probes_b_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, self.file_max_write),
            )
            create_ds("times", (1,), maxshape=(self.file_max_write,))

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as file:
            for k, v in list(data.items()):
                dset = file[k]
                if k.startswith("times"):
                    dset.resize((self.file_write_nb,))
                    dset[-1] = v
                else:
                    dset.resize((self.probes_nb_loc, self.file_write_nb))
                    dset[:, -1] = v

    def _add_probes_data_to_dict(self, data_dict, key):
        """Probes fields and append data to a dict object"""
        data_dict[f"probes_{key}_loc"] = self.sim.state.get_var(key)[
            self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
        ]

    def _online_save(self):
        """Prepares data and writes to file"""
        if self.probes_nb_loc > 0:
            tsim = self.sim.time_stepping.t
            if (
                tsim + 1e-15
            ) // self.period_save > self.t_last_save // self.period_save:
                # if max write number is reached, init new file
                if self.file_write_nb >= self.file_max_write:
                    self.file_nb += 1
                    self.file_write_nb = 0
                    self._init_new_file()
                # get data from probes
                data = {"times": self.sim.time_stepping.t}
                data["times"] = self.sim.time_stepping.t
                for key in self.keys_fields:
                    self._add_probes_data_to_dict(data, key)
                # write to file
                self.file_write_nb += 1
                self._write_to_file(data)
                self.t_last_save = tsim

    def load_time_series(self, key="b", region=None, tmin=0, tmax=None):
        """load time series from files"""
        key = f"probes_{key}_loc"
        if region is None:
            region = self.probes_region
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        xmin, xmax, ymin, ymax, zmin, zmax = region

        # get ranks
        files = list(self.path_dir.glob("rank*"))
        ranks = [int(f.name[4:8]) for f in files]

        # load series
        series = []
        series_times = np.array([])
        for rank in ranks:
            files = list(self.path_dir.glob(f"rank{rank:04}*"))

            data = []
            times = []

            for filename in files:
                with h5py.File(filename, "r") as file:
                    probes_x = file["probes_x_loc"][:]
                    probes_y = file["probes_y_loc"][:]
                    probes_z = file["probes_z_loc"][:]
                    probes_times = file["times"][:]

                    cond_region = (
                        (probes_x > xmin)
                        & (probes_x < xmax)
                        & (probes_y > ymin)
                        & (probes_y < ymax)
                        & (probes_z > zmin)
                        & (probes_z < zmax)
                    )
                    cond_times = (probes_times > tmin) & (probes_times < tmax)

                    data += [file[key][cond_region, cond_times]]
                    times += [probes_times[cond_times]]

            series += [np.concatenate(data, axis=1)]
            series_times = np.concatenate(times)

        result = {key: series, "times": series_times}
        return result
