"""Script to configure before Fluidsim setup.
Custom paths for MPI and FFTW libraries and shared objects are managed here.

Provides
--------
MPI4PY : bool
    True if mpi4py installed and can be imported

FFTW3 : bool
    True if FFTW3 library is available

"""

import os
import sys
import subprocess
import shlex
from ctypes.util import find_library
from distutils.util import strtobool
from configparser import ConfigParser
from logging import ERROR, INFO, DEBUG

from transonic.dist import get_logger

FLUIDDYN_DEBUG = os.environ.get("FLUIDDYN_DEBUG", False)
PARALLEL_COMPILE = not FLUIDDYN_DEBUG

if "egg_info" in sys.argv:
    level = ERROR
elif FLUIDDYN_DEBUG:
    level = DEBUG
else:
    level = INFO

logger = get_logger("fluidsim")
logger.setLevel(level)


TRANSONIC_BACKEND = os.environ.get("FLUIDSIM_TRANSONIC_BACKEND", "pythran")


if "DISABLE_PYTHRAN" in os.environ:
    DISABLE_PYTHRAN = strtobool(os.environ["DISABLE_PYTHRAN"])

    if (
        "FLUIDSIM_TRANSONIC_BACKEND" in os.environ
        and DISABLE_PYTHRAN
        and TRANSONIC_BACKEND == "pythran"
    ):
        raise ValueError

    if DISABLE_PYTHRAN:
        TRANSONIC_BACKEND = "python"


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


FFTW3 = check_avail_library("fftw3")


def get_default_config():
    """Generate default configuration."""
    config = ConfigParser()
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

    # handle environ (variables) in config
    if "environ" in config_dict:
        os.environ.update(config_dict["environ"])

    return config_dict


if __name__ == "__main__":
    make_site_cfg_default_file()
