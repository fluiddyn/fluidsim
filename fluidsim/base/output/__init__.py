"""Output (:mod:`fluidsim.base.output`)
=============================================

Provides:

.. autosummary::
   :toctree:

   base
   prob_dens_func
   spectra
   phys_fields
   spatial_means
   time_signalsK
   spatial_means
   time_signalsK
   increments
   print_stdout
   spect_energy_budget


.. autoclass:: OutputBase
   :members:
   :private-members:

.. autoclass:: OutputBasePseudoSpectral
   :members:
   :private-members:

"""
from __future__ import division

import os
from glob import glob
import argparse

import h5netcdf

import fluiddyn.output

from .base import OutputBase, OutputBasePseudoSpectral


__all__ = ['OutputBase', 'OutputBasePseudoSpectral']


def create_description_xmf_files(path=None):
    """Create description xmf files for Paraview"""

    if path is None:
        path = os.getcwd()

    if os.path.isdir(path):
        paths = glob(path + '/state_phys*.nc')
    else:
        paths = glob(path)

    if len(paths) == 0:
        raise ValueError('No file corresponds to this path.')

    for path in paths:
        path_out = path.split('.nc')[0] + '.xmf'

        base_name = os.path.basename(path)

        with h5netcdf.File(path) as f:
            ndim = 3

            nx = f['/info_simul/params/oper'].attrs['nx']
            Lx = f['/info_simul/params/oper'].attrs['Lx']
            deltax = Lx/nx
            try:
                ny = f['/info_simul/params/oper'].attrs['ny']
                Ly = f['/info_simul/params/oper'].attrs['Ly']
                deltay = Ly/ny
            except KeyError:
                ndim = 1
            try:
                nz = f['/info_simul/params/oper'].attrs['nz']
                Lz = f['/info_simul/params/oper'].attrs['Lz']
                deltaz = Lz/nz
            except KeyError:
                ndim = 2

            keys = list(f['/state_phys'].keys())

        if ndim == 1:
            geometry_type = 'Origin_Dx'
            dims_data = '{}'.format(nx)
            origins = '0'
            deltaxs = '{}'.format(deltax)
        elif ndim == 2:
            geometry_type = 'Origin_DxDy'
            dims_data = '{} {}'.format(ny, nx)
            origins = '0 0'
            deltaxs = '{} {}'.format(deltay, deltax)
        elif ndim == 3:
            geometry_type = 'Origin_DxDyDz'
            dims_data = '{} {} {}'.format(nz, ny, nx)
            origins = '0 0 0'
            deltaxs = '{} {} {}'.format(deltaz, deltay, deltax)

        txt = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf>
  <Domain>
    <Grid Name="my_uniform_grid" GridType="Uniform">
      <Time Value="0." />
      <Topology TopologyType="{ndim}DCoRectMesh" Dimensions="{dims_data}">
      </Topology>
      <Geometry GeometryType="{geometry_type}">
        <DataItem Dimensions="{ndim}" NumberType="Float" Format="XML">
        {origins}
        </DataItem>
        <DataItem Dimensions="{ndim}" NumberType="Float" Format="XML">
        {deltaxs}
        </DataItem>
      </Geometry>
""".format(ndim=ndim, geometry_type=geometry_type, dims_data=dims_data,
           origins=origins, deltaxs=deltaxs)

        for key in keys:
            txt += """
      <Attribute Name="{key}" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{dims_data}" NumberType="Float" Format="HDF">
          {file_name}:/state_phys/{key}
        </DataItem>
      </Attribute>
""".format(key=key, dims_data=dims_data, file_name=base_name)

        txt += """    </Grid>
  </Domain>
</Xdmf>
"""

        with open(path_out, 'w') as f:
            f.write(txt)


def create_description_xmf_file(path=None):
    """Create description xmf file for Paraview"""

    if path is None:
        path = os.getcwd()

    if os.path.isdir(path):
        paths = glob(path + '/state_phys*.nc')
    else:
        paths = glob(path)

    if len(paths) == 0:
        raise ValueError('No file corresponds to this path.')

    path_out = 'states_phys.xmf'

    paths.sort()
    path = paths[0]

    with h5netcdf.File(path) as f:
        ndim = 3

        nx = f['/info_simul/params/oper'].attrs['nx']
        Lx = f['/info_simul/params/oper'].attrs['Lx']
        deltax = Lx/nx
        try:
            ny = f['/info_simul/params/oper'].attrs['ny']
            Ly = f['/info_simul/params/oper'].attrs['Ly']
            deltay = Ly/ny
        except KeyError:
            ndim = 1
        try:
            nz = f['/info_simul/params/oper'].attrs['nz']
            Lz = f['/info_simul/params/oper'].attrs['Lz']
            deltaz = Lz/nz
        except KeyError:
            ndim = 2

        keys = list(f['/state_phys'].keys())

    if ndim == 1:
        geometry_type = 'Origin_Dx'
        dims_data = '{}'.format(nx)
        origins = '0'
        deltaxs = '{}'.format(deltax)
    elif ndim == 2:
        geometry_type = 'Origin_DxDy'
        dims_data = '{} {}'.format(ny, nx)
        origins = '0 0'
        deltaxs = '{} {}'.format(deltay, deltax)
    elif ndim == 3:
        geometry_type = 'Origin_DxDyDz'
        dims_data = '{} {} {}'.format(nz, ny, nx)
        origins = '0 0 0'
        deltaxs = '{} {} {}'.format(deltaz, deltay, deltax)

    txt = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf>
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection"
          CollectionType="Temporal">
    """

    for path in paths:
        base_name = os.path.basename(path)

        with h5netcdf.File(path) as f:
            time = f['state_phys'].attrs['time']

        txt += """
    <Grid Name="my_uniform_grid" GridType="Uniform">
      <Time Value="{time:.9f}" />
      <Topology TopologyType="{ndim}DCoRectMesh" Dimensions="{dims_data}">
      </Topology>
      <Geometry GeometryType="{geometry_type}">
        <DataItem Dimensions="{ndim}" NumberType="Float" Format="XML">
        {origins}
        </DataItem>
        <DataItem Dimensions="{ndim}" NumberType="Float" Format="XML">
        {deltaxs}
        </DataItem>
      </Geometry>
""".format(time=float(time), ndim=ndim, geometry_type=geometry_type,
           dims_data=dims_data, origins=origins, deltaxs=deltaxs)

        for key in keys:
            txt += """
      <Attribute Name="{key}" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{dims_data}" NumberType="Float" Format="HDF">
          {file_name}:/state_phys/{key}
        </DataItem>
      </Attribute>
""".format(key=key, dims_data=dims_data, file_name=base_name)

        txt += '    </Grid>'

    txt += """
    </Grid>
  </Domain>
</Xdmf>
"""

    with open(path_out, 'w') as f:
        f.write(txt)


def run():

    parser = argparse.ArgumentParser(
        prog='fluidsim-create-xml-description',
        description=create_description_xmf_file.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'str_files', nargs='?', default=os.getcwd(),
        help='str indicating which file has to be dump.',
        type=str)

    args = parser.parse_args()

    create_description_xmf_file(args.str_files)
