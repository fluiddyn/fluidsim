"""
FrequencySpectra (:mod:`fluidsim.solvers.ns3d.output.temporal_spectra`)
==============================================================================


Provides:

.. autoclass:: TemporalSpectra3D
   :members:
   :private-members:

.. autoclass:: TemporalSpectra2D
   :members:
   :private-members:

"""

from pathlib import Path
from logging import warn

from math import pi
import numpy as np
from scipy import signal
import h5py
from rich.progress import Progress, track

from fluiddyn.util import mpi
from fluidsim.base.output.base import SpecificOutput
from fluidsim.base.output.spatiotemporal_spectra import (
    filter_tmins_paths,
    get_arange_minmax,
)


class TemporalSpectra3D(SpecificOutput):
    """
    Computes the temporal spectra.
    """

    _tag = "temporal_spectra"
    nb_dim = 3

    @classmethod
    def _complete_params_with_default(cls, params):
        tag = "temporal_spectra"

        params.output.periods_save._set_attrib(tag, 0)

        attribs = {
            "probes_deltax": 0.1,  # m
            "probes_deltay": 0.1,  # m
            "probes_region": None,  # m
            "file_max_size": 10.0,  # MB
            "SAVE_AS_FLOAT32": True,
        }

        if cls.nb_dim == 3:
            attribs["probes_deltaz"] = 0.1  # m

        params.output._set_child(
            tag,
            attribs=attribs,
        )

        params.output.temporal_spectra._set_doc(
            """
            probes_deltax, probes_deltay and probes_deltaz: float (default: 0.1)

                Probes spacing in the x, y and z directions (in params.oper.Li unit).

            probes_region: tuple (default:None)

                Boundaries of the region in the simulation domain were probes are set.

                probes_region = (xmin, xmax, ymin, ymax, zmin, zmax), in params.oper.Lx unit.

                If None, set to the whole simulation domain.

            file_max_size: float (default: 10.0)

                Maximum size of one time series file, in megabytes.

            SAVE_AS_FLOAT32: bool (default: True)

                If set to False, probes data is saved as float64.

                Warning : saving as float64 reduces digital noise at high frequencies, but double the size of the output!

            """
        )

    def __init__(self, output):
        params = output.sim.params
        try:
            params_tspec = params.output.temporal_spectra
        except AttributeError:
            warn(
                "Cannot initialize temporal spectra output because "
                "`params` does not contain parameters for this class."
            )
            return

        super().__init__(
            output,
            period_save=params.output.periods_save.temporal_spectra,
        )

        oper = self.sim.oper

        # Parameters
        self.probes_deltax = params_tspec.probes_deltax
        self.probes_deltay = params_tspec.probes_deltay
        if self.nb_dim == 3:
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
            if self.nb_dim == 3:
                xmin, xmax, ymin, ymax, zmin, zmax = self.probes_region
            else:
                xmin, xmax, ymin, ymax = self.probes_region
                zmin, zmax = 0.0, 0.0
        else:
            xmin = ymin = zmin = 0.0
            xmax = oper.Lx
            ymax = oper.Ly

            if self.nb_dim == 3:
                zmax = oper.Lz
                self.probes_region = xmin, xmax, ymin, ymax, zmin, zmax
            else:
                zmax = 0.0
                self.probes_region = xmin, xmax, ymin, ymax

        self.file_max_size = params_tspec.file_max_size
        self.SAVE_AS_FLOAT32 = params_tspec.SAVE_AS_FLOAT32

        if self.SAVE_AS_FLOAT32:
            size_1_number = 4e-6
            self.datatype = np.float32
        else:
            size_1_number = 8e-6
            self.datatype = np.float64

        if self.nb_dim == 3:
            X, Y, Z = oper.get_XYZ_loc()
        else:
            X = oper.X
            Y = oper.Y

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

        if self.nb_dim == 3:
            self.probes_deltaz = max(
                oper.deltaz, oper.deltaz * round(self.probes_deltaz / oper.deltaz)
            )
            zmin = oper.deltaz * round(zmin / oper.deltaz)
        else:
            self.probes_deltaz = None
            zmin = 0.0

        # make sure probes region is not empty, and xmax is included
        xmax += 1e-15
        ymax += 1e-15
        zmax += 1e-15

        # global probes coordinates
        self.probes_x_seq = np.arange(xmin, xmax, self.probes_deltax)
        self.probes_y_seq = np.arange(ymin, ymax, self.probes_deltay)
        self.probes_z_seq = np.arange(zmin, zmax, self.probes_deltaz)

        # data directory
        if mpi.rank == 0:
            self.path_dir.mkdir(exist_ok=True)
        if mpi.nb_proc > 1:
            mpi.comm.barrier()

        # check for existing files
        paths = sorted(self.path_dir.glob("rank*.h5"))
        if paths:
            # check values in files
            with h5py.File(paths[0], "r") as file:
                if file.attrs["nb_proc"] != mpi.nb_proc:
                    raise ValueError("process number is different from files")
                if not (
                    np.allclose(file["probes_x_seq"][:], self.probes_x_seq)
                    and np.allclose(file["probes_y_seq"][:], self.probes_y_seq)
                    and np.allclose(file["probes_z_seq"][:], self.probes_z_seq)
                ):
                    raise ValueError("probes position are different from files")
            # init from files
            paths = [p for p in paths if p.name.startswith(f"rank{mpi.rank:05}")]
            if paths:
                self.path_file = paths[-1]
                with h5py.File(self.path_file, "r") as file:
                    self.index_file = file.attrs["index_file"]
                    self.probes_x_loc = file["probes_x_loc"][:]
                    self.probes_y_loc = file["probes_y_loc"][:]
                    self.probes_z_loc = file["probes_z_loc"][:]
                    self.probes_ix_loc = file["probes_ix_loc"][:]
                    self.probes_iy_loc = file["probes_iy_loc"][:]
                    self.probes_iz_loc = file["probes_iz_loc"][:]
                    self.probes_nb_loc = self.probes_x_loc.size
                    self.number_times_in_file = file["times"].size
                    self.t_last_save = file["times"][-1]
            else:
                # no probes in proc
                self.path_file = None
                self.index_file = 0
                self.number_times_in_file = 0
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
            self.probes_x_loc = self.probes_x_seq[
                (self.probes_x_seq >= X.min()) & (self.probes_x_seq <= X.max())
            ]
            self.probes_y_loc = self.probes_y_seq[
                (self.probes_y_seq >= Y.min()) & (self.probes_y_seq <= Y.max())
            ]

            if self.nb_dim == 3:
                self.probes_z_loc = self.probes_z_seq[
                    (self.probes_z_seq >= Z.min())
                    & (self.probes_z_seq <= Z.max())
                ]
            else:
                self.probes_z_loc = self.probes_z_seq

            if self.nb_dim == 2:
                assert self.probes_z_loc.size == 1

            self.probes_nb_loc = (
                self.probes_x_loc.size
                * self.probes_y_loc.size
                * self.probes_z_loc.size
            )

            # local probes indices
            self.probes_ix_loc = np.empty(self.probes_nb_loc, dtype=int)
            self.probes_iy_loc = np.empty_like(self.probes_ix_loc)
            self.probes_iz_loc = np.zeros_like(self.probes_ix_loc)
            probe_i = 0
            for probe_x in self.probes_x_loc:
                for probe_y in self.probes_y_loc:
                    for probe_z in self.probes_z_loc:
                        probe_ix = int((probe_x - X.min()) / oper.deltax)
                        probe_iy = int((probe_y - Y.min()) / oper.deltay)
                        self.probes_ix_loc[probe_i] = probe_ix
                        self.probes_iy_loc[probe_i] = probe_iy
                        if self.nb_dim == 3:
                            probe_iz = int((probe_z - Z.min()) / oper.deltaz)
                            self.probes_iz_loc[probe_i] = probe_iz
                        probe_i += 1

            self.probes_x_loc = self._get_data_probe_from_field(X)
            self.probes_y_loc = self._get_data_probe_from_field(Y)
            if self.nb_dim == 3:
                self.probes_z_loc = self._get_data_probe_from_field(Z)

            # initialize files
            self.index_file = 0
            self.number_times_in_file = 0
            self.t_last_save = -self.period_save
            if self.probes_nb_loc > 0:
                self._init_new_file(tmin_file=self.sim.time_stepping.t)

        # size of a single write: nb_fields * probes_nb_loc + time
        probes_write_size = (
            len(self.keys_fields) * self.probes_nb_loc + 1
        ) * size_1_number
        self.max_number_times_in_file = int(
            self.file_max_size / probes_write_size
        )

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self, tmin_file=None):
        """Initializes a new file"""
        if tmin_file is not None:
            # max number of digits = int(log10(t_end)) + 1
            # add .3f precision = 4 additional characters
            # +2 by anticipation of potential restarts
            str_width = int(np.log10(self.sim.params.time_stepping.t_end)) + 7
            ind_str = f"tmin{tmin_file:0{str_width}.3f}"
        else:
            ind_str = f"file{self.index_file:04}"
        self.path_file = self.path_dir / f"rank{mpi.rank:05}_{ind_str}.h5"
        with h5py.File(self.path_file, "w") as file:
            file.attrs["nb_proc"] = mpi.nb_proc
            file.attrs["index_file"] = self.index_file
            file.attrs["period_save"] = self.period_save
            create_ds = file.create_dataset
            create_ds("probes_x_seq", data=self.probes_x_seq)
            create_ds("probes_y_seq", data=self.probes_y_seq)
            create_ds("probes_z_seq", data=self.probes_z_seq)

            create_ds("probes_x_loc", data=self.probes_x_loc)
            create_ds("probes_y_loc", data=self.probes_y_loc)
            create_ds("probes_z_loc", data=self.probes_z_loc)

            create_ds("probes_ix_loc", data=self.probes_ix_loc)
            create_ds("probes_iy_loc", data=self.probes_iy_loc)
            create_ds("probes_iz_loc", data=self.probes_iz_loc)

            create_ds("times", (1,), maxshape=(None,))

            for key in self.keys_fields:
                create_ds(
                    f"probes_{key}_loc",
                    (self.probes_nb_loc, 1),
                    maxshape=(self.probes_nb_loc, None),
                    dtype=self.datatype,
                )

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as file:
            for k, v in data.items():
                dset = file[k]
                if k.startswith("times"):
                    dset.resize((self.number_times_in_file,))
                    if self.SAVE_AS_FLOAT32:
                        v = np.array(v, dtype="float32")
                    dset[-1] = v
                else:
                    dset.resize((self.probes_nb_loc, self.number_times_in_file))
                    if self.SAVE_AS_FLOAT32:
                        v = v.astype("float32")
                    dset[:, -1] = v

    def _get_data_probe_from_field(self, field):
        return field[self.probes_iz_loc, self.probes_iy_loc, self.probes_ix_loc]

    def _add_probes_data_to_dict(self, data_dict, key):
        """Probes fields and append data to a dict object"""
        data_dict[f"probes_{key}_loc"] = self._get_data_probe_from_field(
            self.sim.state.get_var(key)
        )

    def _online_save(self):
        """Prepares data and writes to file"""
        if self.probes_nb_loc > 0:
            tsim = self.sim.time_stepping.t
            if (
                tsim + 1e-15
            ) // self.period_save > self.t_last_save // self.period_save:
                # if max write number is reached, init new file
                if self.number_times_in_file >= self.max_number_times_in_file:
                    self.index_file += 1
                    self.number_times_in_file = 0
                    self._init_new_file(tmin_file=self.sim.time_stepping.t)
                # get data from probes
                data = {"times": self.sim.time_stepping.t}
                data["times"] = self.sim.time_stepping.t
                for key in self.keys_fields:
                    self._add_probes_data_to_dict(data, key)
                # write to file
                self.number_times_in_file += 1
                self._write_to_file(data)
                self.t_last_save = tsim

    def load_time_series(
        self, keys=None, region=None, tmin=0, tmax=None, dtype=None
    ):
        """load time series from files"""
        if keys is None:
            keys = self.keys_fields
        if region is None:
            region = self._get_default_region()
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        if self.nb_dim == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = region
        else:
            xmin, xmax, ymin, ymax = region

        # get ranks
        paths = sorted(self.path_dir.glob("rank*.h5"))
        ranks = sorted({int(p.name[4:9]) for p in paths})

        # get times from the files of first rank
        print(f"load times series...")
        paths_1st_rank = [
            p for p in paths if p.name.startswith(f"rank{ranks[0]:05}")
        ]

        if dtype is None:
            with h5py.File(paths_1st_rank[0], "r") as file:
                dtype = file[f"probes_{keys[0]}_loc"].dtype

        # get list of useful files, from tmin
        tmins_files = np.array([float(p.name[14:-3]) for p in paths_1st_rank])
        tmins_files, paths_1st_rank = filter_tmins_paths(
            tmin, tmins_files, paths_1st_rank
        )

        with Progress() as progress:
            npaths = len(paths_1st_rank)
            task_files = progress.add_task(
                "Getting times from rank 0...", total=npaths
            )

            times = []
            for ip, path in enumerate(paths_1st_rank):
                with h5py.File(path, "r") as file:
                    if tmins_files[ip] > tmax:
                        progress.update(task_files, completed=npaths)
                        break
                    times_file = file["times"][:]
                    cond_times = (times_file >= tmin) & (times_file <= tmax)
                    times.append(times_file[cond_times])
                    progress.update(task_files, advance=1)

        times = np.concatenate(times)

        tmin = times.min()
        tmax = times.max()
        print(f"tmin={tmin:8.6g}, tmax={tmax:8.6g}, nit={times.size}")

        # load series
        series = {f"probes_{k}_loc": [] for k in keys}
        with Progress() as progress:
            task_ranks = progress.add_task("Rearranging...", total=len(ranks))
            task_files = progress.add_task("Rank 00000...", total=npaths)
            # loop on all files, rank by rank
            for rank in ranks:
                paths_rank = [
                    p for p in paths if p.name.startswith(f"rank{rank:05}")
                ]

                # get list of useful files, from tmin
                tmins_files = np.array([float(p.name[14:-3]) for p in paths_rank])
                tmins_files, paths_rank = filter_tmins_paths(
                    tmin, tmins_files, paths_rank
                )

                npaths = len(paths_rank)
                progress.update(
                    task_files,
                    description=f"Rank {rank:05}...",
                    total=npaths,
                    completed=0,
                )

                # for a given rank, paths are sorted by time
                data = {f"probes_{k}_loc": [] for k in keys}
                for path_file in paths_rank:
                    # break after the last useful file
                    if tmins_files[ip] > tmax:
                        progress.update(task_files, completed=npaths)
                        break

                    with h5py.File(path_file, "r") as file:
                        probes_x = file["probes_x_loc"][:]
                        probes_y = file["probes_y_loc"][:]
                        probes_z = file["probes_z_loc"][:]

                        cond_region = (
                            (probes_x >= xmin)
                            & (probes_x <= xmax)
                            & (probes_y >= ymin)
                            & (probes_y <= ymax)
                        )

                        if self.nb_dim == 3:
                            cond_region = (
                                cond_region
                                & (probes_z >= zmin)
                                & (probes_z <= zmax)
                            )

                        cond_region = np.where(cond_region)[0]

                        for key in keys:
                            skey = f"probes_{key}_loc"
                            tmp = file[skey][cond_region, :]

                            times_file = file["times"][:]
                            its_file = get_arange_minmax(times_file, tmin, tmax)
                            data[skey].append(tmp[:, its_file])

                    # update rich task
                    progress.update(task_files, advance=1)

                for key in keys:
                    skey = f"probes_{key}_loc"
                    series[skey].append(np.concatenate(data[skey], axis=1))

                # update rich task
                progress.update(task_ranks, advance=1)

        series["times"] = times
        return series

    def _compute_spectrum(self, data):
        if not hasattr(self, "f_sample"):
            paths = sorted(self.path_dir.glob("rank*.h5"))
            with h5py.File(paths[0], "r") as file:
                self.f_sample = 1.0 / file.attrs["period_save"]
            self.domega = 2 * pi * self.f_sample / data.shape[-1]

        # TODO: I'm not sure if detrend=False is good in prod, but it's much
        # better for testing
        freq, spectrum = signal.periodogram(
            data,
            fs=self.f_sample,
            scaling="spectrum",
            detrend=False,
            return_onesided=False,
        )
        return freq, spectrum / self.domega

    def compute_spectra(self, region=None, tmin=0, tmax=None, dtype=None):
        """compute temporal spectra from files"""
        if region is None:
            region = self._get_default_region()
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        spectra = {"region": region, "tmin": tmin, "tmax": tmax}

        # load data
        series = self.load_time_series(
            region=region, tmin=tmin, tmax=tmax, dtype=dtype
        )

        # compute periodograms and average
        for key in series.keys():
            if key.startswith("probes_"):
                freq, spectrum = self._compute_spectrum(
                    np.concatenate(series[key])
                )
                spectrum = spectrum.mean(0)
                # get one-sided spectra
                nomega = freq.size // 2 + 1
                spectrum_onesided = np.zeros(nomega)
                spectrum_onesided[0] = spectrum[0]
                spectrum_onesided[1:] = (
                    spectrum[1:nomega] + spectrum[-1:-nomega:-1]
                )

                spectra["spectrum_" + key[7:-4]] = spectrum_onesided

        spectra["omegas"] = 2 * pi * freq[:nomega]

        # total kinetic energy
        if self.nb_dim == 3:
            spectra["spectrum_K"] = 0.5 * (
                spectra["spectrum_vx"]
                + spectra["spectrum_vy"]
                + spectra["spectrum_vz"]
            )
        else:
            spectra["spectrum_K"] = 0.5 * (
                spectra["spectrum_ux"] + spectra["spectrum_uy"]
            )

        # potential energy
        try:
            N = self.sim.params.N
            spectra["spectrum_A"] = 0.5 / N**2 * spectra["spectrum_b"]
        except AttributeError:
            pass

        return spectra

    def _get_default_region(self):
        p_oper = self.sim.params.oper
        return (0, p_oper.Lx, 0, p_oper.Ly, 0, p_oper.Lz)

    def plot_spectra(
        self, key=None, region=None, tmin=0, tmax=None, dtype=None, xscale="log"
    ):
        """plot temporal spectra from files"""
        if key is None:
            key = self.keys_fields[0]
        if region is None:
            region = self._get_default_region()
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # load or compute spectra
        path_file = self._get_path_saved_spectra(region, tmin, tmax, dtype)
        if path_file.exists():
            spectra = self.load_spectra(
                region=region, tmin=tmin, tmax=tmax, dtype=dtype
            )
        else:
            spectra = self.save_spectra(
                region=region, tmin=tmin, tmax=tmax, dtype=dtype
            )

        # plot
        fig, ax = self.output.figure_axe()
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("spectrum")
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        ax.set_title(
            f"{key} temporal spectrum (tmin={tmin:.3f}, tmax={tmax:.3f})\n"
            + self.output.summary_simul
        )

        # specific to strat
        try:
            N = self.sim.params.N
        except AttributeError:
            omegas = spectra["omegas"]
            ax.plot(
                spectra["omegas"],
                spectra["spectrum_" + key],
                "k",
                linewidth=2,
            )
        else:
            # kinetic/potential decomposition
            EK = spectra["spectrum_K"]
            EA = spectra["spectrum_A"]
            omegas = spectra["omegas"] / N
            EKN = EK[abs(omegas - 1).argmin()]  # value @N

            ax.plot(omegas, EK, "r", linewidth=2, label=r"$E_K$")
            ax.plot(omegas, EA, "b", linewidth=2, label=r"$E_A$")
            ax.set_title(
                f"kinetic/potential energy spectrum (tmin={tmin:.3f}, tmax={tmax:.3f})\n"
                + self.output.summary_simul
            )

            # resonant modes
            if self.nb_dim == 3:
                aspect_ratio = self.sim.oper.Lx / self.sim.oper.Lz
            else:
                aspect_ratio = self.sim.oper.Lx / self.sim.oper.Ly

            def modes(nx, nz):
                return np.sqrt(nx**2 / (nx**2 + aspect_ratio**2 * nz**2))

            nxs = np.arange(1, 11)
            modes_nz1 = modes(nxs, 1)
            modes_nz2 = modes(nxs, 2)
            modes_y = np.full_like(modes_nz1, fill_value=100 * EKN)

            ax.plot(modes_nz1, modes_y, "o", label="modes $n_z=1$")
            ax.plot(modes_nz2, modes_y * 3, "o", label="modes $n_z=2$")

            # omega^-2 scaling
            omegas_scaling = np.arange(0.4, 1 + 1e-15, 0.01)
            scaling_y = EKN * omegas_scaling**-2

            ax.plot(omegas_scaling, scaling_y, "k--")

            # eye guide @N
            ymin = EKN / 10
            _, ymax = ax.get_ylim()
            ax.vlines(1, ymin, ymax, linestyle="dotted")

            # eye guide @omega_f (specific to watu_coriolis)
            if self.sim.params.forcing.type == "watu_coriolis":
                omega_f = self.sim.params.forcing.watu_coriolis.omega_f
                ax.vlines(omega_f / N, ymin, ymax, linestyle="dotted")

            ax.set_xlabel(r"$\omega/N$")
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(omegas[1], 1.5)

            ax.legend()

    def save_data_as_phys_fields(self, delta_index_times=1):
        """load temporal data and save as phys_fields array"""

        # path to saving directory
        path_dir_save = self.path_dir / "phys_fields"
        path_dir_save.mkdir(exist_ok=True)

        # get ranks
        paths = sorted(self.path_dir.glob("rank*.h5"))
        ranks = sorted({int(p.name[4:9]) for p in paths})

        # get times and probes positions from the files of first rank
        print("save probe data as 3D arrays")

        paths_1st_rank = [
            p for p in paths if p.name.startswith(f"rank{ranks[0]:05}")
        ]

        with h5py.File(paths_1st_rank[0], "r") as file:
            probes_x_seq = file["probes_x_seq"][:]
            probes_y_seq = file["probes_y_seq"][:]
            probes_z_seq = file["probes_z_seq"][:]

        times = []
        for path in paths_1st_rank:
            with h5py.File(path, "r") as file:
                times.append(file["times"][:])
        times = np.concatenate(times)[::delta_index_times]

        print(f"tmin={times.min():8.6g}, tmax={times.max():8.6g}")

        # time string width
        # digits for integer part : int(log10(t)) + 1
        # add 2 zeros, coma and 3 decimals : + 6
        width = int(np.log10(times.max())) + 7

        # probes positions
        xmin = probes_x_seq.min()
        deltax = probes_x_seq[1] - xmin
        ymin = probes_y_seq.min()
        deltay = probes_y_seq[1] - ymin
        zmin = probes_z_seq.min()
        deltaz = probes_z_seq[1] - zmin
        probes_Z, probes_Y, probes_X = np.meshgrid(
            probes_z_seq, probes_y_seq, probes_x_seq, indexing="ij"
        )

        # loop on times
        for time in track(times, description="Rearranging..."):
            # initialize arrays
            dict_arrays = {k: np.empty_like(probes_X) for k in self.keys_fields}

            # loop on ranks
            for rank in ranks:
                for path_file in paths:
                    if not path_file.name.startswith(f"rank{rank:05}"):
                        continue
                    with h5py.File(path_file, "r") as file:
                        # check if the file contains the time we're looking for
                        tmin_file, tmax_file = file["times"][[0, -1]]
                        if (time < tmin_file) or (time > tmax_file):
                            continue

                        # time index
                        it = np.where(file["times"][:] == time)[0]

                        # get global probes indices
                        coord_loc = file["probes_x_loc"][:]
                        ix = np.rint((coord_loc - xmin) / deltax).astype("int")
                        coord_loc = file["probes_y_loc"][:]
                        iy = np.rint((coord_loc - ymin) / deltay).astype("int")
                        coord_loc = file["probes_z_loc"][:]
                        iz = np.rint((coord_loc - zmin) / deltaz).astype("int")

                        # load data at time t for all keys_fields
                        for key in self.keys_fields:
                            dict_arrays[key][iz, iy, ix] = file[
                                f"probes_{key}_loc"
                            ][:, it].transpose()

                        # stop opening files when we've reached the right one
                        break

            # save fields into a new file
            path_file_save = (
                path_dir_save / f"probes_fields_t{time:0{width}.3f}.h5"
            )
            with h5py.File(path_file_save, "w") as file:
                create_ds = file.create_dataset
                # probes coordinates
                create_ds("x", data=probes_X)
                create_ds("y", data=probes_Y)
                create_ds("z", data=probes_Z)
                # physical fields
                for k, v in dict_arrays.items():
                    create_ds(k, data=v)
                # sim info
                self.sim.info._save_as_hdf5(hdf5_parent=file)

    def _get_path_saved_spectra(self, region, tmin, tmax, dtype):
        base = (
            "periodogram_" + "_".join(str(x) for x in region) + f"_{tmin}_{tmax}"
        )
        if dtype is not None:
            base += f"_{dtype}"
        return self.path_dir / (base + ".h5")

    def save_spectra(self, region=None, tmin=0, tmax=None, dtype=None):
        """compute temporal spectra from files"""
        if region is None:
            region = self._get_default_region()
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        spectra = self.compute_spectra(
            region=region, tmin=tmin, tmax=tmax, dtype=dtype
        )

        path_file = self._get_path_saved_spectra(region, tmin, tmax, dtype)
        with h5py.File(path_file, "w") as file:
            file.attrs["region"] = region
            file.attrs["tmin"] = tmin
            file.attrs["tmax"] = tmax
            for key, val in spectra.items():
                file.create_dataset(key, data=val)

        return spectra

    def load_spectra(self, region=None, tmin=0, tmax=None, dtype=None):
        """load temporal spectra from file"""
        if region is None:
            region = self._get_default_region()
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        spectra = {"region": region, "tmin": tmin, "tmax": tmax}

        path_file = self._get_path_saved_spectra(region, tmin, tmax, dtype)
        with h5py.File(path_file, "r") as file:
            for key in file.keys():
                spectra[key] = file[key][...]

        return spectra


class TemporalSpectra2D(TemporalSpectra3D):
    nb_dim = 2

    def _get_data_probe_from_field(self, field):
        return field[self.probes_iy_loc, self.probes_ix_loc]

    def save_data_as_phys_fields(self, delta_index_times=1):
        """load temporal data and save as phys_fields array"""
        raise NotImplementedError

    def _get_default_region(self):
        p_oper = self.sim.params.oper
        return (0, p_oper.Lx, 0, p_oper.Ly)
