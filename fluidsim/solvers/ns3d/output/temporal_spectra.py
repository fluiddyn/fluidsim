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
import h5py
import numpy as np

from scipy import signal
from pathlib import Path

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
                "file_max_size": 1e-2,  # MB
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

        # Parameters
        self.probes_deltax = params_tspec.probes_deltax
        self.probes_deltay = params_tspec.probes_deltay
        self.probes_deltaz = params_tspec.probes_deltaz
        self.period_save = params.output.periods_save.frequency_spectra

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

        self.has_to_save = bool(params.output.periods_save.temporal_spectra)

        # operator
        oper = self.sim.oper
        X, Y, Z = oper.get_XYZ_loc()

        # global probes coordinates
        probes_x_seq = np.arange(probes_xmin, probes_xmax, probes_deltax)
        probes_y_seq = np.arange(probes_ymin, probes_ymax, probes_deltay)
        probes_z_seq = np.arange(probes_zmin, probes_zmax, probes_deltaz)

        # local probes coordinates
        test = (probes_x_seq >= X.min()) & (probes_x_seq <= X.max())
        probes_x_loc = probes_x_seq[test]
        test = (probes_y_seq >= Y.min()) & (probes_y_seq <= Y.max())
        probes_y_loc = probes_y_seq[test]
        test = (probes_z_seq >= Z.min()) & (probes_z_seq <= Z.max())
        probes_z_loc = probes_z_seq[test]

        probes_nb_loc = probes_x_loc.size * probes_y_loc.size * probes_z_loc.size

        # local probes indices
        self.probes_ix_loc = np.empty(probes_nb_loc, dtype=int)
        self.probes_iy_loc = np.empty_like(probes_ix_loc)
        self.probes_iz_loc = np.empty_like(probes_ix_loc)
        probe_i = 0
        for probe_x in probes_x_loc:
            for probe_y in probes_y_loc:
                for probe_z in probes_z_loc:
                    probe_iz, probe_iy, probe_ix = np.where(
                        (abs(X - probe_x) <= oper.deltax / 2)
                        & (abs(Y - probe_y) <= oper.deltay / 2)
                        & (abs(Z - probe_z) <= oper.deltaz / 2)
                    )
                    self.probes_ix_loc[probe_i] = probe_ix
                    self.probes_iy_loc[probe_i] = probe_iy
                    self.probes_iz_loc[probe_i] = probe_iz
                    probe_i += 1

        # files max size
        probes_write_size = (
            4 * probes_nb_loc * 8e-6
        )  # (vx,vy,vz,b) * probes * float64 / MB
        self.file_max_write = 1
        if probes_write_size > 0:
            self.file_max_write = int(self.file_max_size / probes_write_size)

        # create directory
        dir_name = "probes"
        self.path_dir = Path(self.sim.output.path_run) / dir_name
        self.path_dir.mkdir(exist_ok=True)

        # initialize file
        self.file_nb = 0
        self.file_write_nb = 0
        self._init_new_file()

    def _init_files(self, arrays_1st_time=None):
        # we don't want to do anything when this function is called.
        pass

    def _init_new_file(self):
        """Initializes a new file"""
        filename = f"rank{mpi.rank}_file{self.file_nb}.hdf5"
        self.path_file = os.path.join(self.path_dir, filename)
        with h5py.File(self.path_file, "w") as f:
            f.create_dataset("probes_ix_loc", data=self.probes_ix_loc)
            f.create_dataset("probes_iy_loc", data=self.probes_iy_loc)
            f.create_dataset("probes_iz_loc", data=self.probes_iz_loc)
            f.create_dataset(
                "probes_vx_loc",
                (probes_nb_loc, 1),
                maxshape=(probes_nb_loc, self.file_max_write),
            )
            f.create_dataset(
                "probes_vy_loc",
                (probes_nb_loc, 1),
                maxshape=(probes_nb_loc, self.file_max_write),
            )
            f.create_dataset(
                "probes_vz_loc",
                (probes_nb_loc, 1),
                maxshape=(probes_nb_loc, self.file_max_write),
            )
            f.create_dataset(
                "probes_b_loc",
                (probes_nb_loc, 1),
                maxshape=(probes_nb_loc, self.file_max_write),
            )
            f.create_dataset("times", (1,), maxshape=(self.file_max_write,))

    def _write_to_file(self, data):
        """Writes a file with the temporal data"""
        with h5py.File(self.path_file, "a") as f:
            dset = f["probes_vx_loc"]
            dset.resize((probes_nb_loc, file_write_nb))
            dset[:, -1] = data["vx"]
            dset = f["probes_vy_loc"]
            dset.resize((probes_nb_loc, file_write_nb))
            dset[:, -1] = data["vy"]
            dset = f["probes_vz_loc"]
            dset.resize((probes_nb_loc, file_write_nb))
            dset[:, -1] = data["vz"]
            dset = f["times"]
            dset.resize((file_write_nb,))
            dset[-1] = data["time"]

    def _online_save(self):
        """Prepares data and writes to file"""
        # if max write number is reached, init new file
        if self.file_write_nb >= self.file_max_write:
            self.file_nb += 1
            self.file_write_nb = 0
            self._init_new_file()
        # get data from probes
        data = {}
        data["time"] = self.sim.time_stepping.t
        temp = self.sim.state.get_var("vx")
        data["vx"] = temp[
            self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
        ]
        temp = self.sim.state.get_var("vy")
        data["vy"] = temp[
            self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
        ]
        temp = self.sim.state.get_var("vz")
        data["vz"] = temp[
            self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
        ]
        temp = self.sim.state.get_var("b")
        data["b"] = temp[
            self.probes_ix_loc, self.probes_iy_loc, self.probes_iz_loc
        ]
        # write to file
        self._write_to_file(data)
