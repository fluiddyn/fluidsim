"""Script to configure before Fluidsim setup.
Custom paths for MPI and FFTW libraries and shared objects are managed here.

Provides
--------
MPI4PY : bool
    True if mpi4py installed and can be imported

FFTW3 : bool
    True if FFTW3 library is available

"""

from __future__ import print_function

import os
import sys
import subprocess
import shlex
from ctypes.util import find_library

import multiprocessing
from distutils.ccompiler import CCompiler

try:
    from Cython.Distutils import build_ext
    from Cython.Distutils.extension import Extension
except ImportError:
    from distutils.command.build_ext import build_ext
    from setuptools import Extension
try:  # python 3
    from configparser import ConfigParser
except ImportError:  # python 2.7
    from ConfigParser import ConfigParser

try:
    import colorlog as logging

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.ColoredFormatter("%(log_color)s%(levelname)s: %(message)s")
    )
except ImportError:
    import logging

    handler = logging.StreamHandler()


logger = logging.getLogger("fluidsim")
logger.addHandler(handler)
logger.setLevel(20)


try:
    from concurrent.futures import ThreadPoolExecutor as Pool
except ImportError:
    from multiprocessing.pool import ThreadPool as LegacyPool

    # To ensure the with statement works. Required for some older 2.7.x releases
    class Pool(LegacyPool):

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            self.join()

    logger.warn(
        "Falling back to multiprocessing.pool.ThreadPool\n"
        "    pip install futures\n"
        "to use concurrent.futures Python 2.7 backport.\n"
    )

PARALLEL_COMPILE = True


def check_avail_library(library_name):
    """Check if a shared library is available.

    Parameters
    ----------
    library_name : str

    Returns
    -------
    bool

    """
    if find_library(library_name) is not None:
        return True
    elif sys.platform.startswith("linux"):
        try:
            libraries = subprocess.check_output(shlex.split("/sbin/ldconfig -p"))
        except subprocess.CalledProcessError:
            libraries = []

            library_name = "lib" + library_name

        try:
            library_name = library_name.encode("utf8")
        except AttributeError:
            pass

        return library_name in libraries
    else:
        return False


on_rtd = os.environ.get("READTHEDOCS")


if on_rtd:
    MPI4PY = False
else:
    try:
        import mpi4py
    except ImportError:
        MPI4PY = False
        logger.info("ImportError of mpi4py: no mpi extensions will be built.")
    else:
        MPI4PY = True
        CC = os.getenv("CC", "mpicc")
        logger.info(
            "Compiling Cython extensions with the compiler/wrapper: " + CC
        )


FFTW3 = check_avail_library("fftw3")


def build_extensions(self):
    """Function to monkey-patch
    distutils.command.build_ext.build_ext.build_extensions

    """
    self.check_extensions_list(self.extensions)
    try:
        self.compiler.compiler_so.remove("-Wstrict-prototypes")
    except (AttributeError, ValueError):
        pass

    for ext in self.extensions:
        try:
            ext.sources = self.cython_sources(ext.sources, ext)
        except AttributeError:
            pass

    try:
        num_jobs = int(os.environ["FLUIDDYN_NUM_PROCS_BUILD"])
    except KeyError:
        try:
            num_jobs = os.cpu_count()
        except AttributeError:
            num_jobs = multiprocessing.cpu_count()

    cython_extensions = []
    pythran_extensions = []
    for ext in self.extensions:
        if isinstance(ext, Extension):
            cython_extensions.append(ext)
        else:
            pythran_extensions.append(ext)

    def names(exts):
        return [ext.name for ext in exts]

    with Pool(num_jobs) as pool:
        logger.info("Start build_extension: {}".format(names(cython_extensions)))
        pool.map(self.build_extension, cython_extensions)

    logger.info("Stop build_extension: {}".format(names(cython_extensions)))

    with Pool(num_jobs) as pool:
        logger.info("Start build_extension: {}".format(names(pythran_extensions)))
        pool.map(self.build_extension, pythran_extensions)

    logger.info("Stop build_extension: {}".format(names(pythran_extensions)))


def compile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    """Function to monkey-patch distutils.ccompiler.CCompiler"""
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Return *all* object filenames, not just the ones we just built.
    return objects


def monkeypatch_parallel_build():
    """Monkey-patch to compile in parallel."""
    if PARALLEL_COMPILE:
        build_ext.build_extensions = build_extensions
        CCompiler.compile = compile


def get_default_config():
    """Generate default configuration."""
    config = ConfigParser()
    section = "exclude_pythran"
    config.add_section(section)
    exclude_pythran = (
        "fluidsim.solvers.plate2d",
        "fluidsim.solvers.plate2d.output",
        "fluidsim.solvers.sw1l",
        "fluidsim.solvers.sw1l.output",
        "fluidsim.solvers.ns2d.strat",
        "fluidsim.solvers.ns2d.bouss",
    )
    for excluded in exclude_pythran:
        config.set(section, excluded, "True")

    config.add_section("environ")

    return config


def make_site_cfg_default_file():
    """Write the default configuration to site.cfg.default."""

    config = get_default_config()

    with open("site.cfg.default", "w") as configfile:
        config.write(configfile)
        configfile.write(
            """
## Uncomment and specify the following options to modify compilation of
## extensions.

## To modify compiler used to build Cython extensions:
# MPICXX =
# CC =
# LDSHARED =

## To modify compiler used to build Pythran extensions (or alternatively,
## set ~/.pythranrc. A word of caution --- the pythranrc approach may result in
## race condition for setting and unsetting compilers for pythran > 0.8.6):
# CXX =

## To modify target architecture while building Pythran extensions
## Useful when cross-compiling. See whether it is required by comparing:
## 	gcc -march=native -Q --help=target
## 	gcc -march=$CARCH -Q --help=target
# CARCH =
"""
        )


def get_config():
    """Check for site-specific configuration file saved as either:

    1. site.cfg in source directory, or
    2. $HOME/.fluidsim-site.cfg

    and read if found, else revert to default configuration.

    Returns
    -------
    dict

    """
    config = get_default_config()

    user_dir = "~user" if sys.platform == "win32" else "~"
    configfile_user = os.path.expanduser(
        os.path.join(user_dir, ".fluidsim-site.cfg")
    )

    for configfile in ("site.cfg", configfile_user):
        if os.path.exists(configfile):
            logger.info("Parsing " + configfile)
            config.read(configfile)
            break
    else:
        logger.info("Using default configuration.")
        logger.info(
            "Copy site.cfg.default -> site.cfg or $HOME/.fluidsim-site.cfg "
            "to specify site specific libraries."
        )

    config_dict = {}
    for section in config.sections():

        section_dict = {}
        config_dict[section] = section_dict
        for option in config.options(section):
            if section == "environ":
                option = option.upper()
                value = config.get(section, option)
            else:
                value = config.getboolean(section, option)

            section_dict[option] = value

    return config_dict


if __name__ == "__main__":
    make_site_cfg_default_file()
