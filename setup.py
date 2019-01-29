import os
import sys
from pathlib import Path

from time import time
from runpy import run_path

from setuptools.dist import Distribution
from setuptools import setup, find_packages

try:
    from Cython.Distutils.extension import Extension
    from Cython.Compiler import Options as CythonOptions

    has_cython = True
    ext_source = "pyx"
except ImportError:
    from setuptools import Extension

    has_cython = False
    ext_source = "c"

here = Path(__file__).parent.absolute()

try:
    from setup_config import MPI4PY, FFTW3, logger
except ImportError:
    # needed when there is already a module with the same name imported.
    setup_config = run_path(here / "setup_config.py")
    MPI4PY = setup_config["MPI4PY"]
    FFTW3 = setup_config["FFTW3"]
    logger = setup_config["logger"]

try:
    from setup_build import FluidSimBuildExt
except ImportError:
    # needed when there is already a module with the same name imported.
    FluidSimBuildExt = run_path(here / "setup_build.py")["FluidSimBuildExt"]

time_start = time()


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


# Get the long description from the relevant file
with open(os.path.join(here, "README.rst")) as file:
    long_description = file.read()
lines = long_description.splitlines(True)
long_description = "".join(lines[14:])

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
    "transonic>=0.1.8",
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


def install_setup_requires():
    dist = Distribution()
    # Honor setup.cfg's options.
    dist.parse_config_files(ignore_option_errors=True)
    if dist.setup_requires:
        dist.fetch_build_eggs(dist.setup_requires)


def transonize():

    from transonic.dist import make_backend_files

    paths = [
        "fluidsim/base/time_stepping/pseudo_spect.py",
        "fluidsim/base/output/increments.py",
        "fluidsim/operators/operators2d.py",
        "fluidsim/operators/operators3d.py",
        "fluidsim/solvers/ns2d/solver.py",
    ]
    make_backend_files(
        [here / path for path in paths],
        mocked_modules=(
            "psutil",
            "h5py",
            "matplotlib",
            "matplotlib.pyplot",
            "fluiddyn",
            "fluiddyn.io",
            "fluiddyn.util",
            "fluiddyn.util.paramcontainer",
            "fluiddyn.util.mpi",
            "fluiddyn.output",
            "fluiddyn.calcul",
            "fluiddyn.calcul.setofvariables",
            "fluidfft.fft2d.operators",
            "fluidfft.fft3d.operators",
        ),
    )


def create_pythran_extensions():
    import numpy as np
    from transonic.dist import init_pythran_extensions

    compile_arch = os.getenv("CARCH", "native")
    extensions = init_pythran_extensions(
        "fluidsim",
        include_dirs=np.get_include(),
        compile_args=("-O3", "-march={}".format(compile_arch), "-DUSE_XSIMD"),
    )
    return extensions


def create_extensions():
    if "egg_info" in sys.argv:
        return []

    logger.info("Running fluidsim setup.py on platform " + sys.platform)
    logger.info(__about__)

    install_setup_requires()

    transonize()

    logger.info("Importing mpi4py: {}".format(MPI4PY))

    define_macros = []
    if has_cython and os.getenv("TOXENV") is not None:
        cython_defaults = CythonOptions.get_directive_defaults()
        cython_defaults["linetrace"] = True
        define_macros.append(("CYTHON_TRACE_NOGIL", "1"))

    import numpy as np

    path_sources = "fluidsim/base/time_stepping"
    ext_cyfunc = Extension(
        "fluidsim.base.time_stepping.pseudo_spect_cy",
        include_dirs=[path_sources, np.get_include()],
        libraries=["m"],
        library_dirs=[],
        sources=[path_sources + "/pseudo_spect_cy." + ext_source],
        define_macros=define_macros,
    )

    ext_modules = [ext_cyfunc]

    logger.info(
        "The following extensions could be built if necessary:\n"
        + "".join([ext.name + "\n" for ext in ext_modules])
    )

    ext_modules.extend(create_pythran_extensions())

    return ext_modules


setup(
    version=__version__,
    long_description=long_description,
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
        "Programming Language :: Cython",
        "Programming Language :: C",
    ],
    packages=find_packages(exclude=["doc", "examples"]),
    install_requires=install_requires,
    cmdclass={"build_ext": FluidSimBuildExt},
    ext_modules=create_extensions(),
    entry_points={"console_scripts": console_scripts},
)

logger.info("Setup completed in {:.3f} seconds.".format(time() - time_start))
