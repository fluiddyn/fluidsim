"""Module to configure before Fluidsim setup.
Custom paths for MPI and FFTW libraries and shared objects are managed here.

Provides
--------
MPI4PY : bool
    True if mpi4py installed and can be imported

FFTW3 : bool
    True if FFTW3 library is available

FFTW3MPI : bool
    True if FFTW3-MPI library is available

dict_ldd : dict
    list of flags to use while building shared objects (*.so)

dict_lib : dict
    list of paths where necessary libraries can be found (eg: libmpi.so)

dict_inc : dict
    list of paths to include where necessary header files can be found (eg:
    mpi-compat.h, fftw3.h)

"""

from __future__ import print_function

import os
import sys
import subprocess


def check_avail_library(library_name):
    try:
        libraries = subprocess.check_output('/sbin/ldconfig -p', shell=True)
    except subprocess.CalledProcessError:
        libraries = []

    if sys.platform != 'win32':
        library_name = 'lib' + library_name

    try:
        library_name = library_name.encode('utf8')
    except AttributeError:
        pass

    return library_name in libraries


def find_library_dirs(args, library_dirs=None, debug=True, skip=True):
    """
    Takes care of non-standard library directories, instead of using LDFLAGS.
    Set `skip` as `True` if you do not want to use this.
    """

    if library_dirs is None:
        library_dirs = []

    if skip:
        # print('* Skipping search for additional LDFLAGS: ', args)
        return library_dirs

    for library_name in args:
        libso = "'lib"+library_name+".so'"
        filter1 = " | grep " + libso
        filter2 = " | awk -F'=> ' '{print $2}'"
        filter3 = " | awk -F" + libso + " '{print $1}'"
        try:
            cmd = 'readlink -f $(/sbin/ldconfig -p' + \
                  filter1 + filter2 + ')' + filter3
            if debug:
                print(cmd)

            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            dirs = proc.communicate()[0]
            for item in dirs.split('\n'):
                if item != '':
                    library_dirs.append(item)

        except subprocess.CalledProcessError:
            print('* Cannot extract library directory: ' + library_name)

    default_dirs = ['/usr/lib/', '/usr/lib32/']
    library_dirs = list(set(library_dirs) - set(default_dirs))
    if debug:
        print('LDFLAGS for ', args, ' => ', library_dirs)

    return library_dirs

on_rtd = os.environ.get('READTHEDOCS')

if on_rtd:
    MPI4PY = False
else:
    try:
        import mpi4py
    except ImportError:
        MPI4PY = False
        print('* ImportError of mpi4py: no mpi extensions will be built.')
    else:
        MPI4PY = True
        try:
            cc = os.environ["CC"]
        except KeyError:
            cc = 'mpicc'
            os.environ["CC"] = cc
        print('* Compiling Cython extensions with the compiler/wrapper: ' + cc)

FFTW3 = check_avail_library('fftw3')
FFTW3MPI = check_avail_library('fftw3_mpi')

# Shared libraries used while compiling shared objects
keys = ['mpi', 'fftw']
dict_ldd = dict.fromkeys(keys, [])
dict_lib = dict.fromkeys(keys, [])
dict_inc = dict.fromkeys(keys, [])

if MPI4PY:
    if os.environ["CC"] not in ('mpicc', 'cc'):
        dict_ldd['mpi'] = ['mpi']

    dict_lib['mpi'] = find_library_dirs(['mpi'])
    here = os.path.abspath(os.path.dirname(__file__))
    dict_inc['mpi'] = [mpi4py.get_include(), here + '/include']

if FFTW3 or FFTW3MPI:
    print('* Compiling FFTW extensions with the fftw3.h and libfftw3.so')
    dict_ldd['fftw'] = ['fftw3']
    if FFTW3MPI:
        print('  ... and also fftw3_mpi.h and libfftw3_mpi.so')
        dict_ldd['fftw'].append('fftw3_mpi')

    try:
        dict_lib['fftw'].append(os.environ['FFTW_DIR'])
        dict_inc['fftw'].append(os.environ['FFTW_INC'])
    except KeyError:
        if sys.platform == 'win32':
            if MPI4PY:
                raise ValueError(
                    'We have to work on this case with MPI4PY on Windows...')
            dict_lib['fftw'].append(r'c:\Prog\fftw-3.3.4-dll64')
            dict_inc['fftw'].append(r'c:\Prog\fftw-3.3.4-dll64')

        else:
            dict_lib['fftw'].extend(find_library_dirs('fftw3'))
            dict_inc['fftw'] = []
