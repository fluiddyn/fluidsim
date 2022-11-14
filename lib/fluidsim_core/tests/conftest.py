import sys
from shutil import copyfile

import pytest

import numpy as np

import pymech

name_solver = "testing"


@pytest.fixture
def path_sim(tmp_path):

    path_dir = tmp_path / "session_00"
    path_dir.mkdir(exist_ok=True, parents=True)

    nx = 2
    ny = 4
    nz = 6
    nx_elem = ny_elem = nz_elem = 2

    hexa_data = pymech.core.HexaData(
        ndim=3,
        nel=nx_elem * ny_elem * nz_elem,
        lr1=(nx, ny, nz),
        var=(3, 3, 1, 1, 2),
    )
    hexa_data.wdsz = 8
    hexa_data.istep = 0
    hexa_data.endian = sys.byteorder

    x1d = np.linspace(0, 1, nx)
    y1d = np.linspace(0, 1, ny)
    z1d = np.linspace(0, 1, nz)

    y3d, z3d, x3d = np.meshgrid(y1d, z1d, x1d)
    assert y3d.shape == (nz, ny, nx)

    ielem = 0
    for iz_elem in range(nz_elem):
        for iy_elem in range(ny_elem):
            for ix_elem in range(nx_elem):
                elem = hexa_data.elem[ielem]
                ielem += 1
                elem.pos[0] = x3d + ix_elem
                elem.pos[1] = y3d + iy_elem
                elem.pos[2] = z3d + iz_elem
                elem.vel.fill(1)

    time = 2.0
    for it in range(4):
        hexa_data.time = time
        path = path_dir / f"{name_solver}0.f{it:05d}"
        pymech.writenek(path, hexa_data)
        copyfile(path, path.with_name("sts" + path.name))
        time += 0.5

    return path_dir.parent


class Object:
    pass


@pytest.fixture
def false_output(path_sim):
    output = Object()
    output.name_solver = name_solver
    output.sim = Object()
    output.sim.params = Object()
    output.params = Object()
    output.sim.params.output = Object()
    output.sim.oper = Object()
    output.sim.oper.axes = "zyx"
    output.sim.params.output.session_id = 0
    output.path_run = path_sim
    return output
