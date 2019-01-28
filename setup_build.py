import os
from pathlib import Path
from runpy import run_path

from concurrent.futures import ThreadPoolExecutor as Pool
import multiprocessing

try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext

    has_cython = True
    ext_source = "pyx"
except ImportError:
    from setuptools import Extension
    from distutils.command.build_ext import build_ext

    has_cython = False
    ext_source = "c"


try:
    from pythran.dist import PythranExtension, PythranBuildExt
except ImportError:
    PythranBuildExt = object


here = Path(__file__).parent.absolute()

try:
    from setup_config import PARALLEL_COMPILE, logger
except ImportError:
    # needed when there is already a module with the same name imported.
    setup_config = run_path(here / "setup_config.py")
    logger = setup_config["logger"]
    PARALLEL_COMPILE = setup_config["PARALLEL_COMPILE"]


try:
    num_jobs = int(os.environ["FLUIDDYN_NUM_PROCS_BUILD"])
except KeyError:
    num_jobs = multiprocessing.cpu_count()

    try:
        from psutil import virtual_memory
    except ImportError:
        pass
    else:
        avail_memory_in_Go = virtual_memory().available / 1e9
        limit_num_jobs = round(avail_memory_in_Go / 3)
        num_jobs = min(num_jobs, limit_num_jobs)


class FluidSimBuildExt(build_ext, PythranBuildExt):
    def build_extensions(self):
        """Function to monkey-patch
        distutils.command.build_ext.build_ext.build_extensions

        """
        to_be_removed = ["-Wstrict-prototypes"]
        starts_forbiden = ["-axMIC_", "-diag-disable:"]

        self.compiler.compiler_so = [
            key
            for key in self.compiler.compiler_so
            if key not in to_be_removed
            and all([not key.startswith(s) for s in starts_forbiden])
        ]

        if not PARALLEL_COMPILE:
            return super().build_extensions()

        self.check_extensions_list(self.extensions)

        for ext in self.extensions:
            try:
                ext.sources = self.cython_sources(ext.sources, ext)
            except AttributeError:
                pass

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
            logger.info(
                "Start build_extension: {}".format(names(cython_extensions))
            )
            pool.map(self.build_extension, cython_extensions)

        logger.info("Stop build_extension: {}".format(names(cython_extensions)))

        with Pool(num_jobs) as pool:
            logger.info(
                "Start build_extension: {}".format(names(pythran_extensions))
            )
            pool.map(self.build_extension, pythran_extensions)

        logger.info("Stop build_extension: {}".format(names(pythran_extensions)))
