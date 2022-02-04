import os
import sys
from pathlib import Path

from time import time
from runpy import run_path

from setuptools import setup, find_packages

from transonic.dist import make_backend_files, init_transonic_extensions

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

here = Path(__file__).parent.absolute()

sys.path.insert(0, ".")

from setup_configure import FFTW3, logger, TRANSONIC_BACKEND, get_config
from setup_build import FluidSimBuildExt

time_start = time()

# Parse site.cfg or ~/.fluidsim-site.cfg
_ = get_config()

build_dependencies_backends = {
    "pythran": ["pythran>=0.9.7"],
    "cython": ["cython"],
    "python": [],
    "numba": [],
}

if TRANSONIC_BACKEND not in build_dependencies_backends:
    raise ValueError(
        f"FLUIDSIM_TRANSONIC_BACKEND={TRANSONIC_BACKEND} "
        f"not in {list(build_dependencies_backends.keys())}"
    )

setup_requires = []
setup_requires.extend(build_dependencies_backends[TRANSONIC_BACKEND])

# Set the environment variable FLUIDSIM_SETUP_REQUIRES=0 if we need to skip
# setup_requires for any reason.
if os.environ.get("FLUIDSIM_SETUP_REQUIRES", "1") == "0":
    setup_requires = []


def long_description():
    """Get the long description from the relevant file."""
    with open(os.path.join(here, "README.rst")) as readme:
        lines = list(readme)

    idx = lines.index(".. description\n") + 1
    return "".join(lines[idx:])


# Get the version from the relevant file
version_module = here / "fluidsim" / "_version.py"
version_module_core = here / "lib" / "fluidsim_core" / "_version.py"
version_template = here / "fluidsim" / "_version.tpl"
if not version_module.exists() or (
    version_module_core.exists()
    # check modification time
    and version_module.stat().st_mtime
    < max(version_module_core.stat().st_mtime, version_template.stat().st_mtime)
):
    from string import Template

    logger.info("Writing fluidsim/_version.py")
    version_def = version_module_core.read_text()
    tpl = Template(version_template.read_text())
    version_module.write_text(tpl.substitute(version_def=version_def))
else:
    logger.info("Found fluidsim/_version.py")


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
    f"fluidsim-core>={__version__}",
    "h5py",
    "h5netcdf",
    "transonic>=0.4.3",
    "setuptools_scm",
    "xarray",
    "rich",
    "matplotlib>=3.3",
]

if FFTW3:
    from textwrap import dedent
    from warnings import warn

    fft_extras_msg = dedent(
        """
        ---------------------------------------------------------------------

        FFTW was detected, but pyfftw and fluidfft will not be auto-installed
        (which was the case in previous fluidsim versions). To do so, instead
        of:

            pip install fluidsim

        specify "extras":

            pip install "fluidsim[fft]"

        ---------------------------------------------------------------------
    """
    )
    warn(fft_extras_msg)


def transonize():

    paths = [
        "fluidsim/base/time_stepping/pseudo_spect.py",
        "fluidsim/base/output/increments.py",
        "fluidsim/operators/operators2d.py",
        "fluidsim/operators/operators3d.py",
        "fluidsim/solvers/ns2d/solver.py",
        "fluidsim/solvers/ns3d/strat/solver.py",
        "fluidsim/solvers/ns3d/forcing/watu.py",
        "fluidsim/util/mini_oper_modif_resol.py",
        "fluidsim/base/output/spatiotemporal_spectra.py",
        "fluidsim/solvers/ns3d/output/spatiotemporal_spectra.py",
        "fluidsim/solvers/ns2d/output/spatiotemporal_spectra.py",
    ]
    make_backend_files([here / path for path in paths], backend=TRANSONIC_BACKEND)


def create_pythran_extensions():
    import numpy as np

    compile_arch = os.getenv("CARCH", "native")
    extensions = init_transonic_extensions(
        "fluidsim",
        backend=TRANSONIC_BACKEND,
        include_dirs=np.get_include(),
        compile_args=("-O3", f"-march={compile_arch}", "-DUSE_XSIMD"),
    )
    return extensions


def create_extensions():
    if "egg_info" in sys.argv or "dist_info" in sys.argv:
        return []

    logger.info(
        f"Running fluidsim setup.py ({sys.argv[1:]}) "
        f"on platform {sys.platform}\n " + __about__
    )

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
    url="https://foss.heptapod.net/fluiddyn/fluidsim",
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
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["doc", "examples"]),
    setup_requires=setup_requires,
    install_requires=install_requires,
    cmdclass={"build_ext": FluidSimBuildExt},
    ext_modules=create_extensions(),
)

logger.info(f"Setup completed in {time() - time_start:.3f} seconds.")
