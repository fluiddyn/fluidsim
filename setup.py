import os
import sys
from pathlib import Path

from time import time
from runpy import run_path

from setuptools.dist import Distribution
from setuptools import setup, find_packages

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


def install_setup_requires():
    dist = Distribution()
    # Honor setup.cfg's options.
    dist.parse_config_files(ignore_option_errors=True)
    if dist.setup_requires:
        dist.fetch_build_eggs(dist.setup_requires)


install_setup_requires()

here = Path(__file__).parent.absolute()

try:
    from setup_config import FFTW3, logger
except ImportError:
    # needed when there is already a module with the same name imported.
    setup_config = run_path(here / "setup_config.py")
    FFTW3 = setup_config["FFTW3"]
    logger = setup_config["logger"]

try:
    from setup_build import FluidSimBuildExt
except ImportError:
    # needed when there is already a module with the same name imported.
    FluidSimBuildExt = run_path(here / "setup_build.py")["FluidSimBuildExt"]

time_start = time()


def long_description():
    """Get the long description from the relevant file."""
    with open(os.path.join(here, "README.rst")) as readme:
        lines = list(readme)

    idx = lines.index(".. description\n") + 1
    return "".join(lines[idx:])


# Get the version from the relevant file
version = run_path("fluidsim/_version.py")
__version__ = version["__version__"]
__about__ = version["__about__"]

# Get the development status from the version string
if "a" in __version__:
    devstatus = "Development Status :: 3 - Alpha"
elif "b" in __version__:
    devstatus = "Development Status :: 4 - Beta"
else:
    devstatus = "Development Status :: 5 - Production/Stable"

install_requires = [
    "fluiddyn >= 0.3.0",
    "h5py",
    "h5netcdf",
    "transonic>=0.2.0",
    "setuptools_scm",
    "xarray",
]

if FFTW3:
    install_requires.extend(["pyfftw >= 0.10.4", "fluidfft >= 0.2.7"])

console_scripts = [
    "fluidsim = fluidsim.util.console.__main__:run",
    "fluidsim-test = fluidsim.util.testing:run",
    "fluidsim-create-xml-description = fluidsim.base.output:run",
]

for command in ["profile", "bench", "bench-analysis"]:
    console_scripts.append(
        "fluidsim-"
        + command
        + " = fluidsim.util.console.__main__:run_"
        + command.replace("-", "_")
    )


def transonize():

    from transonic.dist import make_backend_files

    paths = [
        "fluidsim/base/time_stepping/pseudo_spect.py",
        "fluidsim/base/output/increments.py",
        "fluidsim/operators/operators2d.py",
        "fluidsim/operators/operators3d.py",
        "fluidsim/solvers/ns2d/solver.py",
    ]
    make_backend_files([here / path for path in paths])


def create_pythran_extensions():
    import numpy as np
    from transonic.dist import init_pythran_extensions

    compile_arch = os.getenv("CARCH", "native")
    extensions = init_pythran_extensions(
        "fluidsim",
        include_dirs=np.get_include(),
        compile_args=("-O3", f"-march={compile_arch}", "-DUSE_XSIMD"),
    )
    return extensions


def create_extensions():
    if "egg_info" in sys.argv:
        return []

    logger.info("Running fluidsim setup.py on platform " + sys.platform)
    logger.info(__about__)

    transonize()

    ext_modules = create_pythran_extensions()

    logger.info(
        "The following extensions could be built if necessary:\n"
        + "".join([ext.name + "\n" for ext in ext_modules])
    )

    return ext_modules


setup(
    version=__version__,
    long_description=long_description(),
    author="Pierre Augier",
    author_email="pierre.augier@legi.cnrs.fr",
    url="https://bitbucket.org/fluiddyn/fluidsim",
    license="CeCILL",
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        devstatus,
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        # actually CeCILL License (GPL compatible license for French laws)
        #
        # Specify the Python versions you support here. In particular,
        # ensure that you indicate whether you support Python 2,
        # Python 3 or both.
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["doc", "examples"]),
    install_requires=install_requires,
    cmdclass={"build_ext": FluidSimBuildExt},
    ext_modules=create_extensions(),
    entry_points={"console_scripts": console_scripts},
)

logger.info("Setup completed in {:.3f} seconds.".format(time() - time_start))
