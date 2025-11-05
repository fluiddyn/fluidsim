from pathlib import Path

import h5netcdf
import h5py


def open_h5_nc(path, mode):
    """Helper to open .h5 or .nc file with the right package"""
    path = Path(path)

    if path.name.endswith(".nc"):
        h5pack = h5netcdf
    else:
        h5pack = h5py
    return h5pack.File(path, mode)
