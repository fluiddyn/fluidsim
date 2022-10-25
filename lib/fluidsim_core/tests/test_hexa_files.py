from dataclasses import dataclass
import sys

import pytest

import numpy as np
import matplotlib.pyplot as plt

import pymech

from fluidsim_core.hexa_files import SetOfPhysFieldFiles, HexaField


@dataclass
class MockEvent:
    inaxes: object
    xdata: float
    ydata: float


name_solver = "cbox"


@pytest.fixture
def path_sim(tmp_path):

    path_dir = tmp_path / "session_00"
    path_dir.mkdir(exist_ok=True, parents=True)

    nx = ny = nz = 2
    nx_elem = ny_elem = nz_elem = 2

    hexa_data = pymech.core.HexaData(
        ndim=3,
        nel=nx_elem * ny_elem * nz_elem,
        lr1=(nx, ny, nz),
        var=(3, 3, 1, 1, 0),
    )
    hexa_data.wdsz = 8
    hexa_data.istep = 0
    hexa_data.endian = sys.byteorder

    x1d = np.linspace(0, 1, nx)
    y1d = np.linspace(0, 1, ny)
    z1d = np.linspace(0, 1, nz)

    y3d, z3d, x3d = np.meshgrid(x1d, y1d, z1d)

    ielem = 0
    for iz_elem in range(nz_elem):
        for iy_elem in range(ny_elem):
            for ix_elem in range(nx_elem):
                elem = hexa_data.elem[ielem]
                ielem += 1
                elem.pos[0] = x3d + ix_elem
                elem.pos[1] = y3d + iy_elem
                elem.pos[2] = z3d + iz_elem

    time = 2.0
    for it in range(4):
        hexa_data.time = time
        pymech.writenek(path_dir / f"{name_solver}0.f{it:05d}", hexa_data)
        time += 0.5

    return path_dir.parent


def test_setoffiles(path_sim):

    hexa_data_loaded = pymech.readnek(
        path_sim / f"session_00/{name_solver}0.f{1:05d}"
    )

    HexaField(hexa_data=hexa_data_loaded, key="vz")
    HexaField(hexa_data=hexa_data_loaded, key="pres", equation=None)

    class Object:
        pass

    output = Object()
    output.name_solver = "cbox"
    output.sim = Object()
    output.sim.params = Object()
    output.sim.params.output = Object()
    output.sim.params.output.session_id = 0

    set_of_files = SetOfPhysFieldFiles(path_sim, output=output)
    hexa_z, time = set_of_files.get_field_to_plot(2, key="z", equation=None)

    set_of_files.get_key_field_to_plot("pressure")

    2.0 * hexa_z + hexa_z

    set_of_files.plot_hexa()

    ax = plt.gca()
    on_move = ax.figure._on_move_hexa
    event = MockEvent(ax, 0.5, 0.5)
    on_move(event)

    set_of_files.get_vector_for_plot(time=3)
    set_of_files.get_dataset_from_time(2.5)
