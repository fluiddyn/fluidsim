from pathlib import Path
from runpy import run_path

try:
    from transonic.dist import ParallelBuildExt
except ImportError:
    from distutils.command.build_ext import build_ext as ParallelBuildExt

here = Path(__file__).parent.absolute()

try:
    from setup_config import PARALLEL_COMPILE, logger
except ImportError:
    # needed when there is already a module with the same name imported.
    setup_config = run_path(here / "setup_config.py")
    logger = setup_config["logger"]
    PARALLEL_COMPILE = setup_config["PARALLEL_COMPILE"]


logger.debug("FluidSimBuildExt base class = {}".format(ParallelBuildExt))


class FluidSimBuildExt(ParallelBuildExt):
    def initialize_options(self):
        super().initialize_options()
        self.logger_name = "fluidsim"
        self.num_jobs_env_var = "FLUIDDYN_NUM_PROCS_BUILD"

    def get_num_jobs(self):
        if PARALLEL_COMPILE:
            return super().get_num_jobs()
        else:
            # Return None which would in turn retain the `self.parallel` in its
            # default value
            return None
