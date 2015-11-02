
from __future__ import print_function

from setuptools import setup, find_packages

try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools import Extension
    from distutils.command import build_ext

import subprocess
import numpy as np

import os
here = os.path.abspath(os.path.dirname(__file__))

import sys
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[0:2] < (3, 2):
    raise RuntimeError("Python version 2.7 or >= 3.2 required.")

# Get the long description from the relevant file
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()
lines = long_description.splitlines(True)
long_description = ''.join(lines[8:])

# Get the version from the relevant file
d = {}
execfile('fluidsim/_version.py', d)
__version__ = d['__version__']

# Get the development status from the version string
if 'a' in __version__:
    devstatus = 'Development Status :: 3 - Alpha'
elif 'b' in __version__:
    devstatus = 'Development Status :: 4 - Beta'
else:
    devstatus = 'Development Status :: 5 - Production/Stable'


try:
    libraries = subprocess.check_output('/sbin/ldconfig -p', shell=True)
except subprocess.CalledProcessError:
    libraries = []


def check_avail_library(library_name):
    if sys.platform != 'win32':
        library_name = 'lib' + library_name

    return library_name in libraries

def find_library_dirs(args, library_dirs=[], debug=False):
    for library_name in args:
        libso = "'lib"+library_name+".so'"
        filter1 = " | grep " + libso 
        filter2 = " | awk -F'=> ' '{print $2}'"
        filter3 = " | awk -F" + libso + " '{print $1}'"
        try:
            cmd = '/sbin/ldconfig -p' + filter1 + filter2 + filter3
            if debug:
                print(cmd)

            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            dirs = proc.communicate()[0]
            for item in dirs.split('\n'):
                if item is not '':
                    library_dirs.append(item)

        except subprocess.CalledProcessError:
            print('Cannot extract library directory: '+library_name)

    default_dirs = ['/usr/lib/','/usr/lib32/']
    library_dirs = list(set(library_dirs) - set(default_dirs))
    print(args,' => ',library_dirs)
    return library_dirs


try:
    import mpi4py
except ImportError:
    MPI4PY = False
    include_dirs_mpi = []
    print('ImportError of mpi4py: no mpi extensions will be built.')
else:
    MPI4PY = True
    os.environ["CC"] = 'mpicc'
    include_dirs_mpi = [
        mpi4py.get_include(),
        here + '/include']

FFTW3 = check_avail_library('fftw3')
FFTW3MPI = check_avail_library('fftw3_mpi')
include_dirs_fftw = []
if FFTW3 or FFTW3MPI:
    try:
        include_dirs_fftw = os.environ['FFTW_INC'] # TODO: Needs to be more general
    except KeyError:
        pass

del libraries


ext_modules = []

if MPI4PY and FFTW3:
    path_sources = 'fluidsim/operators/fft/Sources_fftw2dmpiccy'
    include_dirs = [path_sources, np.get_include()] + include_dirs_mpi + include_dirs_fftw
    libraries = ['mpi', 'fftw3', 'm']
    library_dirs = find_library_dirs(libraries)

    ext_fftw2dmpiccy = Extension(
        'fluidsim.operators.fft.fftw2dmpiccy',
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        sources=[path_sources+'/libcfftw2dmpi.c',
                 path_sources+'/fftw2dmpiccy.pyx'])
    ext_modules.append(ext_fftw2dmpiccy)

if FFTW3:
    path_sources = 'fluidsim/operators/fft/Sources_fftw2dmpicy'
    include_dirs = [path_sources, np.get_include()] + include_dirs_fftw
    libraries = ['m']
    library_dirs = []
    if MPI4PY:
        include_dirs.extend(include_dirs_mpi)
        libraries.append('mpi')
        library_dirs = find_library_dirs(libraries, library_dirs)

    if sys.platform == 'win32':
        if MPI4PY:
            raise ValueError(
                'We have to work on this case with MPI4PY on Windows...')
        fftw_dir = r'c:\Prog\fftw-3.3.4-dll64'
        library_dirs.append(fftw_dir)
        include_dirs.append(fftw_dir)
        libraries.append('libfftw3-3')
    else:
        libraries.append('fftw3')

    if FFTW3MPI:
        libraries.append('fftw3_mpi')
        library_dirs = find_library_dirs(libraries, library_dirs)

        ext_fftw2dmpicy = Extension(
            'fluidsim.operators.fft.fftw2dmpicy',
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            cython_compile_time_env={'MPI4PY': MPI4PY},
            sources=[path_sources+'/fftw2dmpicy.pyx'])
        ext_modules.append(ext_fftw2dmpicy)


path_sources = 'fluidsim/operators/CySources'
include_dirs = [path_sources, np.get_include()]
libraries = ['m']
if MPI4PY:
    include_dirs.extend(include_dirs_mpi)
    libraries.extend(['mpi'])

library_dirs = find_library_dirs(libraries, library_dirs)
ext_operators = Extension(
    'fluidsim.operators.operators',
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    cython_compile_time_env={'MPI4PY': MPI4PY},
    sources=[path_sources+'/operators_cy.pyx'])

ext_misc = Extension(
    'fluidsim.operators.miscellaneous',
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    cython_compile_time_env={'MPI4PY': MPI4PY},
    sources=[path_sources+'/miscellaneous_cy.pyx'])


path_sources = 'fluidsim/base/time_stepping'
ext_cyfunc = Extension(
    'fluidsim.base.time_stepping.pseudo_spect_cy',
    include_dirs=[
        path_sources,
        np.get_include()],
    libraries=['m'],
    library_dirs=[],
    sources=[path_sources+'/pseudo_spect_cy.pyx'])

ext_modules.extend([
    ext_operators,
    ext_misc,
    ext_cyfunc])

print('The following extensions are going to be built if necessary:\n' +
      ''.join([ext.name + '\n' for ext in ext_modules]))


install_requires = ['fluiddyn >= 0.0.10a7']

on_rtd = os.environ.get('READTHEDOCS')
if not on_rtd:
    install_requires += ['h5py']
    if FFTW3:
        install_requires += ['pyfftw']


setup(name='fluidsim',
      version=__version__,
      description=('Framework for studying fluid dynamics with simulations.'),
      long_description=long_description,
      keywords='Fluid dynamics, research',
      author='Pierre Augier',
      author_email='pierre.augier@legi.cnrs.fr',
      url='https://bitbucket.org/fluiddyn/fluidsim',
      license='CeCILL',
      classifiers=[
          # How mature is this project? Common values are
          # 3 - Alpha
          # 4 - Beta
          # 5 - Production/Stable
          devstatus,
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
          # actually CeCILL License (GPL compatible license for French laws)
          #
          # Specify the Python versions you support here. In particular,
          # ensure that you indicate whether you support Python 2,
          # Python 3 or both.
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          # 'Programming Language :: Python :: 3',
          # 'Programming Language :: Python :: 3.3',
          # 'Programming Language :: Python :: 3.4',
          'Programming Language :: Cython',
          'Programming Language :: C',
      ],
      packages=find_packages(exclude=['doc', 'examples']),
      install_requires=install_requires,
      extras_require=dict(
          doc=['Sphinx>=1.1', 'numpydoc'],
          parallel=['mpi4py']),
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules)
