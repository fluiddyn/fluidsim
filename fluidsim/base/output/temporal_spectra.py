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

from math import pi
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
                "probes_region": None,  # m
                "file_max_size": 10.0,  # MB
            },
        )

        params.output.temporal_spectra._set_doc(
            """
            probes_deltax: float (default: 0.1)

                Probes spacing in the x direction, in meters.

            probes_deltay: float (default: 0.1)

                Probes spacing in the y direction, in meters.

            probes_deltaz: float (default: 0.1)

                Probes spacing in the x direction, in meters.

            probes_region: tuple (default:None)

                Boundaries of the region in the simulation domain were probes are set.
                
                probes_region = (xmax, xmin, ymax, ymin, zmax, zmin), in meters. 
                
                If None, set to the whole simulation domain.

            file_max_size: float (default: 10.0)

                Maximum size of one time series file, in megabytes.
            
            """
        )

    def __init__(self, output):
        params = output.sim.params
        params_tspec = params.output.temporal_spectra

        super().__init__(
            output,
            period_save=params.output.periods_save.temporal_spectra,
            has_to_plot_saved=params_tspec.HAS_TO_PLOT_SAVED,
        )

        oper = self.sim.oper

        # Parameters
        self.probes_deltax = params_tspec.probes_deltax
        self.probes_deltay = params_tspec.probes_deltay
        self.probes_deltaz = params_tspec.probes_deltaz
        self.period_save = params.output.periods_save.temporal_spectra

        self.path_dir = Path(self.sim.output.path_run) / "probes"
        self.keys_fields = self.sim.info_solver.classes.State.keys_state_phys

        if not output._has_to_save:
            self.period_save = 0.0
        if self.period_save == 0.0:
            return

        if params_tspec.probes_region is not None:
            self.probes_region = params_tspec.probes_region
            (
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
            ) = self.probes_region
        else:
            xmin = ymin = zmin = 0.0
            xmax = oper.Lx
            ymax = oper.Ly
            zmax = oper.Lz
            self.probes_region = (
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
            )

        self.file_max_size = params_tspec.file_max_size

        X, Y, Z = oper.get_XYZ_loc()

        # round probes positions to gridpoints
        # probes spacing should at least be oper grid spacing
        self.probes_deltax = max(
            oper.deltax, oper.deltax * round(self.probes_deltax / oper.deltax)
        )
        xmin = oper.deltax * round(xmin / oper.deltax)

        self.probes_deltay = max(
            oper.deltay, oper.deltay * round(self.probes_deltay / oper.deltay)
        )
        ymin = oper.deltay * round(ymin / oper.deltay)

        self.probes_deltaz = max(
            oper.deltaz, oper.deltaz * round(self.probes_deltaz / oper.deltaz)
        )
        zmin = oper.deltaz * round(zmin / oper.deltaz)

        # make sure probes region is not empty, and xmax is included
        xmax += 1e-15
        ymax += 1e-15
        zmax += 1e-15

        # global probes coordinates
        self.probes_x_seq = np.arange(xmin, xmax, self.probes_deltax)
        self.probes_y_seq = np.arange(ymin, ymax, self.probes_deltay)
        self.probes_z_seq = np.arange(zmin, zmax, self.probes_deltaz)

        probes_nb_seq = (
            self.probes_x_seq.size
            * self.probes_y_seq.size
            * self.probes_z_seq.size
        )

        # data directory
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)
        if mpi.nb_proc > 1:
            mpi.comm.barrier()

        # check for existing files
        files = sorted(list(self.path_dir.glob(f"rank{mpi.rank:05}*")))
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
            files = [f for f in files if f.name.startswith(f"rank{mpi.rank:05}")]
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
                    self.t_last_save = file["times"][-1]
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
                        probe_ix = int((probe_x - X.min()) / oper.deltax)
                        probe_iy = int((probe_y - Y.min()) / oper.deltay)
                        probe_iz = int((probe_z - Z.min()) / oper.deltaz)
                        self.probes_ix_loc[probe_i] = probe_ix
                        self.probes_iy_loc[probe_i] = probe_iy
                        self.probes_iz_loc[probe_i] = probe_iz
                        probe_i += 1
            self.probes_x_loc = X[
                self.probes_iz_loc, self.probes_iy_loc, self.probes_ix_loc
            ]
            self.probes_y_loc = Y[
                self.probes_iz_loc, self.probes_iy_loc, self.probes_ix_loc
            ]
            self.probes_z_loc = Z[
                self.probes_iz_loc, self.probes_iy_loc, self.probes_ix_loc
            ]

            # initialize files
            self.file_nb = 0
            self.file_write_nb = 0
            self.t_last_save = -self.period_save
            if self.probes_nb_loc > 0:
                self._init_new_file()

        # files max size (float64 = 8e-6 MB)
        # size of init : 3 coords * (x_loc + ix_loc + x_seq)
        probes_init_size = 3 * (2 * self.probes_nb_loc + probes_nb_seq) * 8e-6
        # size of a single write : nb_fields * probes_nb_loc + time
        probes_write_size = (
            len(self.keys_fields) * self.probes_nb_loc + 1
        ) * 8e-6
        self.file_max_write_nb = int(
            (self.file_max_size - probes_init_size) / probes_write_size
        )

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self):
        """Initializes a new file"""
        self.path_file = (
            self.path_dir / f"rank{mpi.rank:05}_file{self.file_nb:04}.hdf5"
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
                maxshape=(self.probes_nb_loc, None),
            )
            create_ds(
                "probes_vy_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, None),
            )
            create_ds(
                "probes_vz_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, None),
            )
            create_ds(
                "probes_b_loc",
                (self.probes_nb_loc, 1),
                maxshape=(self.probes_nb_loc, None),
            )
            create_ds("times", (1,), maxshape=(None,))

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as file:
            for k, v in data.items():
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
            self.probes_iz_loc, self.probes_iy_loc, self.probes_ix_loc
        ]

    def _online_save(self):
        """Prepares data and writes to file"""
        if self.probes_nb_loc > 0:
            tsim = self.sim.time_stepping.t
            if (
                tsim + 1e-15
            ) // self.period_save > self.t_last_save // self.period_save:
                # if max write number is reached, init new file
                if self.file_write_nb >= self.file_max_write_nb:
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
            oper = self.sim.oper
            region = (0, oper.Lx, 0, oper.Ly, 0, oper.Lz)
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        xmin, xmax, ymin, ymax, zmin, zmax = region

        # get ranks
        files = sorted(list(self.path_dir.glob("rank*")))
        ranks = list({int(f.name[4:9]) for f in files})

        # load series
        series = []
        series_times = np.array([])
        for rank in ranks:
            files = sorted(list(self.path_dir.glob(f"rank{rank:05}*")))

            data = []
            times = []

            for filename in files:
                with h5py.File(filename, "r") as file:
                    probes_x = file["probes_x_loc"][:]
                    probes_y = file["probes_y_loc"][:]
                    probes_z = file["probes_z_loc"][:]
                    probes_times = file["times"][:]

                    cond_region = (
                        (probes_x >= xmin)
                        & (probes_x <= xmax)
                        & (probes_y >= ymin)
                        & (probes_y <= ymax)
                        & (probes_z >= zmin)
                        & (probes_z <= zmax)
                    )
                    cond_times = (probes_times >= tmin) & (probes_times <= tmax)

                    temp = file[key][cond_region, :]
                    data += [temp[:, cond_times]]
                    times += [probes_times[cond_times]]

            series += [np.concatenate(data, axis=1)]
            series_times = np.concatenate(times)

        result = {key: series, "times": series_times}
        return result

    def compute_spectra(self, keys=None, region=None, tmin=0, tmax=None):
        """compute temporal spectra from files"""
        if keys is None:
            keys = self.keys_fields
        if region is None:
            oper = self.sim.oper
            region = (0, oper.Lx, 0, oper.Ly, 0, oper.Lz)
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        dict_spectra = {"region": region, "tmin": tmin, "tmax": tmax}

        for key in keys:
            # load data
            data = self.load_time_series(key, region, tmin, tmax)
            series = np.concatenate(data[f"probes_{key}_loc"])
            times = data["times"]

            # get sampling frequency
            f_sample = 1 / np.mean(times[1:] - times[:-1])

            # compute periodograms and average
            freq, spectra = signal.periodogram(series, fs=f_sample)
            dict_spectra["spectra_" + key] = spectra.mean(0)

        dict_spectra["omegas"] = 2 * pi * freq

        return dict_spectra

    def plot_spectra(self, key="b", region=None, tmin=0, tmax=None):
        """plot temporal spectra from files"""
        if region is None:
            oper = self.sim.oper
            region = (0, oper.Lx, 0, oper.Ly, 0, oper.Lz)
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # compute spectra
        dict_spectra = self.compute_spectra(
            keys=[key], region=region, tmin=tmin, tmax=tmax
        )

        # plot
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("spectra " + key)
        ax.set_title(
            f"temporal spectrum (tmin={tmin:.2g}, tmax={tmax:.2g})\n"
            + self.output.summary_simul
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(
            dict_spectra["omegas"],
            dict_spectra["spectra_" + key],
            "k",
            linewidth=2,
        )
